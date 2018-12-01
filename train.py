import os
import shutil
import pickle

from tqdm import tqdm
import numpy as np
from numpy.random import RandomState
#import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, remove_small_holes

import torch
from torch.utils import data
from torchvision import transforms
from torch.autograd import Variable

from model import *

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def jaccard(intersection, union, eps=1e-15):
    return (intersection + eps) / (union - intersection + eps)

def dice(intersection, union, eps=1e-15):
    return (2. * intersection + eps) / (union + eps)

class BCESoftJaccardDice:

    def __init__(self, bce_weight=0.5, mode="dice", eps=1e-15, weight=None):
        self.nll_loss = torch.nn.BCEWithLogitsLoss(weight=weight)
        self.bce_weight = bce_weight
        self.eps = eps
        self.mode = mode

    def __call__(self, outputs, targets):
        loss = self.bce_weight * self.nll_loss(outputs, targets)

        if self.bce_weight < 1.:
            targets = (targets == 1).float()
            outputs = torch.nn.functional.sigmoid(outputs)
            intersection = (outputs * targets).sum()
            union = outputs.sum() + targets.sum()
            if self.mode == "dice":
                score = dice(intersection, union, self.eps)
            elif self.mode == "jaccard":
                score = jaccard(intersection, union, self.eps)
            loss -= (1 - self.bce_weight) * torch.log(score)
        return loss
    
def get_metric(pred, targets):
    batch_size = targets.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = targets[batch].squeeze(1), pred[batch].squeeze(1)
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue

        t = (t == 1).float()
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        m = dice(intersection, union, eps=1e-15)
        metric.append(m)
    return np.mean(metric)

def leastsq(x, y):
    a = np.vstack([x, np.ones(len(x))]).T
    return np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, y))

class Trainer:
    
    def __init__(self, path=None, **kwargs):
        
        if path is not None:
            kwargs = pickle.load(open(path+"/model_params.pickle.dat", "rb"))
            kwargs["pretrained"] = False
            
        self.model_name = kwargs["model_name"]
        self.directory = kwargs["directory"]
        self.path = os.path.join(self.directory, self.model_name)
        self.device_idx = kwargs["device_idx"]
        self.ADAM = kwargs["ADAM"]
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        
        self.cp_counter_loss, self.cp_counter_metric = 0, 0
        self.max_lr = .5
        
        net_init_params = {k:v for k, v in kwargs.items() 
            if k in ["model", "Dropout", "pretrained", "num_classes", "num_filters", "is_deconv"]
        }

        self.initial_model = UnetResNet(**net_init_params)
       
        if kwargs["reset"]:
            try:
                shutil.rmtree(self.path)
            except:
                pass
            os.mkdir(self.path)
            kwargs["reset"] = False
            pickle.dump(kwargs, open(self.path+"/model_params.pickle.dat", "wb"))
        else:
            self.model = self.get_model(self.initial_model)
            if self.ADAM:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
        
    def get_model(self, model):
        model = model.train()
        return model.cuda(self.device_idx)

    def LR_finder(self, dataset, **kwargs):
        
        max_lr = kwargs["max_lr"]
        batch_size = kwargs["batch_size"]
        learning_rate = kwargs["learning_rate"]
        bce_loss_weight = kwargs["bce_loss_weight"]
        loss_growth_trsh = kwargs["loss_growth_trsh"]
        loss_window = kwargs["loss_window"]
        wd = kwargs["wd"]
        alpha = kwargs["alpha"]
        
        torch.cuda.empty_cache()
        dataset.clear_buff()
        self.model = self.get_model(self.initial_model)

        iterations = len(dataset) // batch_size
        it = 0
        lr_mult = (max_lr/learning_rate)**(1/iterations)

        if self.ADAM:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, 
                                        momentum=0.9, nesterov=True)

        #max LR search
        print(" [INFO] Start max. learning rate search... ")
        min_loss, self.lr_finder_losses = (np.inf, learning_rate), [[], []]
        for image, mask, file_id, mask_w in tqdm(data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers=0)):
            image = image.type(torch.FloatTensor).cuda(self.device_idx)

            it += 1
            current_lr = learning_rate * (lr_mult**it)

            y_pred = self.model(Variable(image))

            loss_fn = BCESoftJaccardDice(bce_weight=bce_loss_weight, 
                                         weight=mask_w.cuda(self.device_idx), mode="dice", eps=1.)
            loss = loss_fn(y_pred, Variable(mask.cuda(self.device_idx)))

            optimizer.zero_grad()
            loss.backward()

            #adjust learning rate and weights decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
                if wd:
                    for param in param_group['params']:
                        param.data = param.data.add(-wd * param_group['lr'], param.data)

            optimizer.step()

            if it > 1:
                current_loss = alpha * loss.item() + (1 - alpha) * current_loss
            else:
                current_loss = loss.item()
            
            self.lr_finder_losses[0].append(current_loss)
            self.lr_finder_losses[1].append(current_lr)
            
            if current_loss < min_loss[0]:
                min_loss = (current_loss, current_lr)
                    
            if it >= loss_window:
                if (current_loss - min_loss[0]) / min_loss[0] >= loss_growth_trsh:
                    break
            
        self.max_lr = round(min_loss[1], 5)
        print(" [INFO] max. lr = %.5f " % self.max_lr)
        
    def show_lr_finder_out(self, save_only=False):
        if not save_only:
            plt.show(block=False)
        plt.semilogx(self.lr_finder_losses[1], self.lr_finder_losses[0])
        plt.axvline(self.max_lr, c="gray")
        plt.savefig(self.path + '/lr_finder_out.png')
        
    def fit(self, dataset, dataset_val, **kwargs):

        epoch = kwargs["epoch"]
        learning_rate = kwargs["learning_rate"]
        batch_size = kwargs["batch_size"]
        bce_loss_weight = kwargs["bce_loss_weight"]
        CLR = kwargs["CLR"]
        wd = kwargs["wd"]
        reduce_lr_patience = kwargs["reduce_lr_patience"]
        reduce_lr_factor = kwargs["reduce_lr_factor"]
        max_lr_decay = kwargs["max_lr_decay"]
        early_stop = kwargs["early_stop"]

        torch.cuda.empty_cache()
        self.model = self.get_model(self.initial_model)
        
        if self.ADAM:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, 
                                             momentum=0.9, nesterov=True)

        max_lr = self.max_lr
        iterations = len(dataset) // batch_size
        if abs(CLR) == 1:
            iterations *= epoch
        lr_mult = (max_lr/learning_rate)**(1/iterations)
        
        checkpoint_metric, checkpoint_loss, it, k, cooldown = -np.inf, np.inf, 0, 1, 0
        x = np.arange(early_stop) / (early_stop - 1)
        self.history = {"loss":{"train":[], "test":[]}, "metric":{"train":[], "test":[]}}
        for e in range(epoch):
            torch.cuda.empty_cache()

            if reduce_lr_patience and reduce_lr_factor:
                if not np.isinf(checkpoint_loss):
                    if self.history["loss"]["test"][-1] >= checkpoint_loss:
                        cooldown += 1

                if cooldown == reduce_lr_patience:
                    learning_rate *= reduce_lr_factor; max_lr *= reduce_lr_factor
                    lr_mult = (max_lr/learning_rate)**(1/iterations)
                    cooldown = 0
                    print(" [INFO] Learning rate has been reduced to: %.7f " % learning_rate)
            
            dataset.clear_buff()
            min_train_loss, train_loss, train_metric = np.inf, [], []
            for image, mask, file_id, mask_w in tqdm(data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers=0)):
                image = image.type(torch.FloatTensor).cuda(self.device_idx)

                if abs(CLR):
                    it += 1; exp = it
                    if CLR > 0:
                        exp = iterations*k - it + 1
                    current_rate = learning_rate * (lr_mult**exp)
                    
                    if abs(CLR) > 1:
                        if iterations*k / it == 1: 
                            it = 0; k *= abs(CLR)
                            if max_lr_decay < 1.:
                                max_lr *= max_lr_decay
                            lr_mult = (max_lr/learning_rate)**(1/(iterations*k))

                            #re-init. optimzer to reset internal state
                            if self.ADAM:
                                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=current_rate)
                            else:
                                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=current_rate, 
                                                                 momentum=0.9, nesterov=True)

                y_pred = self.model(Variable(image))

                loss_fn = BCESoftJaccardDice(bce_weight=bce_loss_weight, 
                                             weight=mask_w.cuda(self.device_idx), mode="dice", eps=1.)
                loss = loss_fn(y_pred, Variable(mask.cuda(self.device_idx)))

                self.optimizer.zero_grad()
                loss.backward()
                
                #adjust learning rate and weights decay
                for param_group in self.optimizer.param_groups:
                    try: param_group['lr'] = current_lr
                    except: pass
                    if wd:
                        for param in param_group['params']:
                            param.data = param.data.add(-wd * param_group['lr'], param.data)

                self.optimizer.step()
                if loss.item() < min_train_loss:
                    min_train_loss = loss.item()
                train_loss.append(loss.item())
                train_metric.append(get_metric((y_pred.cpu() > 0.).float(), mask))
                
            del y_pred; del image; del mask_w; del mask; del loss

            dataset_val.clear_buff()
            torch.cuda.empty_cache()
            val_loss, val_metric = [], []
            for image, mask, file_id, mask_w in data.DataLoader(dataset_val, batch_size = batch_size // 2, shuffle = False, num_workers=0):
                image = image.cuda(self.device_idx)

                y_pred = self.model(Variable(image))
                
                loss_fn = BCESoftJaccardDice(bce_weight=bce_loss_weight, 
                                             weight=mask_w.cuda(self.device_idx), mode="dice", eps=1.)
                loss = loss_fn(y_pred, Variable(mask.cuda(self.device_idx)))

                val_loss.append(loss.item())
                val_metric.append(get_metric((y_pred.cpu() > 0.).float(), mask))
                
            del y_pred; del image; del mask_w; del mask; del loss

            train_loss, train_metric, val_loss, val_metric = \
                np.mean(train_loss), np.mean(train_metric), np.mean(val_loss), np.mean(val_metric)
            
            if val_loss < checkpoint_loss:
                save_checkpoint(self.path+'/%s_checkpoint_loss.pth' % (self.model_name), self.model, self.optimizer)
                checkpoint_loss = val_loss

            if val_metric > checkpoint_metric:
                save_checkpoint(self.path+'/%s_checkpoint_metric.pth' % (self.model_name), self.model, self.optimizer)
                checkpoint_metric = val_metric

            self.history["loss"]["train"].append(train_loss)
            self.history["loss"]["test"].append(val_loss)
            self.history["metric"]["train"].append(train_metric)
            self.history["metric"]["test"].append(val_metric)

            message = "Epoch: %d, Train loss: %.3f, Train metric: %.3f, Val loss: %.3f, Val metric: %.3f" % (
                e, train_loss, train_metric, val_loss, val_metric)
            print(message); os.system("echo " + message)
            
            if early_stop and e >= early_stop:
                wlv, _ = leastsq(x, self.history["loss"]["test"][-early_stop:])
                wlt, _ = leastsq(x, self.history["loss"]["train"][-early_stop:])
                wmv, _ = leastsq(x, self.history["metric"]["test"][-early_stop:])
                wmt, _ = leastsq(x, self.history["metric"]["train"][-early_stop:])
                if wlv >= 0 and wlt <= 0 and wmt >= 0 and wmv <= 0:
                    message = " [INFO] Learning stopped at %s# epoch! " % e
                    print(message); os.system("echo " + message)
                    break
                    
            self.current_epoch = e
            save_checkpoint(self.path+'/last_checkpoint.pth', self.model, self.optimizer)
            
        pickle.dump(self.history, open(self.path+'/history.pickle.dat', 'wb'))
        
    def plot_trainer_history(self, mode="metric", save_only=False):
        if not save_only:
            plt.show(block=False)
        plt.plot(self.history[mode]["train"], label="train")
        plt.plot(self.history[mode]["test"], label="val")
        plt.xlabel("epoch")
        plt.ylabel(mode)
        plt.grid(True)
        plt.legend(loc="best")
        plt.savefig(self.path + '/trainer_history.png')
        
    def load_state(self, path=None, mode="metric", load_optimizer=True):
        if load_optimizer: load_optimizer = self.optimizer
        if path is None:            
            path = self.path+'/%s_checkpoint_%s.pth' % (self.model_name, mode)
        load_checkpoint(path, self.model, load_optimizer)

    def predict_crop(self, imgs):
        torch.cuda.empty_cache()
        if imgs.ndim < 4:
            imgs = np.expand_dims(imgs, axis=0)
        l, h, w, c = imgs.shape
        w_n = int(w/h * min(512, h))
        h_n = min(512, h)
        all_predictions = np.empty((l, h_n, w_n, c+1)).astype(np.uint8)
        for i in range(imgs.shape[0]):
            img = self.norm(cv2.resize(imgs[i], (256, 320), interpolation=cv2.INTER_LANCZOS4))
            img = img.unsqueeze_(0)
            img = img.type(torch.FloatTensor).cuda()
            output = torch.nn.functional.sigmoid(self.model(Variable(img)))
            output = output.cpu().data.numpy()
            y_pred = np.squeeze(output[0])
            y_pred = remove_small_holes(remove_small_objects(y_pred > .3))
            y_pred = (y_pred * 255).astype(np.uint8)
            y_pred = cv2.resize(y_pred, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
            #y_pred = cv2.cvtColor(y_pred,cv2.COLOR_GRAY2BGR)
            #y_pred = cv2.bitwise_and(imgs[i], y_pred.astype(np.uint8))
            img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
            _,alpha = cv2.threshold(y_pred.astype(np.uint8),0,255,cv2.THRESH_BINARY)
            b, g, r = cv2.split(imgs[i])
            bgra = [r,g,b, alpha]
            y_pred = cv2.merge(bgra,4)
            y_pred = cv2.resize(y_pred, (w_n, h_n), interpolation=cv2.INTER_LANCZOS4)
            all_predictions[i] = y_pred
        return all_predictions

if __name__ == "__main__":

    train_path = "./data/train/"
    path_images = list(map(
        lambda x: x.split('.')[0],
        filter(lambda x: x.endswith('.jpg'), os.listdir('./data/train/'))))
    prng = RandomState(42)

    path_images *= 3
    prng.shuffle(path_images)
    train_split = int(len(path_images)*.8)
    train_images, val_images = path_images[:train_split], path_images[train_split:]

    dataset = DatasetProcessor(train_path, train_images, as_torch_tensor=True, augmentations=True, mask_weight=True)
    dataset_val = DatasetProcessor(train_path, val_images, as_torch_tensor=True, augmentations=True, mask_weight=True)

    params = {
        #LR_finder
        "batch_size":20,
        "max_lr":.5,
        "loss_window":10, 
        "loss_growth_trsh":.5,
        "alpha":0.1,
        
        #fit
        "wd":0.,
        "early_stop":10,
        "max_lr_decay":.8,
        "epoch":200,
        "learning_rate":1e-4,
        "bce_loss_weight":0.5,
        "reduce_lr_patience":0,
        "reduce_lr_factor":0,
        "CLR":2
    }

    model_type = "resnet50"

    model_params = {
        "directory":"./data/",
        "model_type":model_type,
        "model_name":"%s_05BCE_no_CLR_200e_SGD_weighted_mask" % model_type,
        "Dropout":.4,
        "device_idx":0,
        "pretrained":True,
        "num_classes":1,
        "num_filters":64,
        "is_deconv":True,
        "reset":True,
        "ADAM":True
    }

    trainer = Trainer(**model_params)
    trainer.show_lr_finder_out(save_only=True)
    trainer.fit(dataset, dataset_val, **params)
    trainer.plot_trainer_history(mode="loss", save_only=True)
    trainer.plot_trainer_history(mode="metric", save_only=True)