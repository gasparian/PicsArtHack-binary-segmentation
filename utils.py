from __future__ import print_function, division
import os
import string
import itertools
import pickle

from skimage.morphology import remove_small_objects, remove_small_holes
import cv2
from tqdm import tqdm
import numpy as np

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    ShiftScaleRotate,
    RandomBrightness
)


import torch
from torchvision import transforms
from torch.utils import data
from torch.autograd import Variable

from model import *

class DatasetProcessor(data.Dataset):
    
    def __init__(self, root_path, file_list, is_test=False, as_torch_tensor=True, augmentations=False, mask_weight=True):
        self.is_test = is_test
        self.mask_weight = mask_weight
        self.root_path = root_path
        self.file_list = file_list
        self.as_torch_tensor = as_torch_tensor
        self.augmentations = augmentations
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        self.been = []
        
    def clear_buff(self):
        self.been = []
    
    def __len__(self):
        return len(self.file_list)

    def transform(self, image, mask):
        aug = Compose([
            HorizontalFlip(p=0.9),
            RandomBrightness(p=.5,limit=0.3),
            RandomContrast(p=.5,limit=0.3),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, 
                             p=0.7,  border_mode=0, interpolation=4)
        ])
        
        augmented = aug(image=image, mask=mask)
        return augmented['image'], augmented['mask']
    
    def get_mask_weight(self, mask):
        mask_ = cv2.erode(mask, kernel=np.ones((8,8),np.uint8), iterations=1)
        mask_ = mask-mask_
        return mask_ + 1
    
    def __getitem__(self, index):
        
        file_id = index
        if type(index) != str:
            file_id = self.file_list[index]
        
        image_folder = self.root_path
        image_path = os.path.join(image_folder, file_id + ".jpg")
        
        mask_folder = self.root_path[:-1] + "_mask/"
        mask_path = os.path.join(mask_folder, file_id + ".png")
        
        if self.as_torch_tensor:
                    
            if not self.is_test:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(mask_path))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                
                #resize to 320x256
                image = cv2.resize(image, (256, 320), interpolation=cv2.INTER_LANCZOS4)
                mask = cv2.resize(mask, (256, 320), interpolation=cv2.INTER_LANCZOS4)
                
                if self.augmentations:
                    if file_id not in self.been:
                        self.been.append(file_id)
                    else:
                        image, mask = self.transform(image, mask)
                    
                mask = mask // 255
                mask = mask[:, :, np.newaxis]
                if self.mask_weight:
                    mask_w = self.get_mask_weight(np.squeeze(mask))
                else: 
                    mask_w = np.ones((mask.shape[:-1]))
                mask_w = mask_w[:, :, np.newaxis]
                    
                mask = torch.from_numpy(np.transpose(mask, (2, 0, 1)).astype('float32'))
                mask_w = torch.from_numpy(np.transpose(mask_w, (2, 0, 1)).astype('float32'))
                image = self.norm(image)
                return image, mask, mask_w

            else:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (256, 320), interpolation=cv2.INTER_LANCZOS4)
                image = self.norm(image)
                return image
            
        else:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image, dtype=np.uint8)
            if not self.is_test:
                mask = cv2.imread(str(mask_path))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                if self.augmentations:
                    if file_id not in self.been:
                        self.been.append(file_id)
                    else:
                        image, mask = self.transform(image, mask)
                return image, mask
            
            else:
                if self.augmentations:
                    if file_id not in self.been:
                        self.been.append(file_id)
                    else:
                        image = self.transform(image)
                return image

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer, cpu):
    if cpu:
        state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def jaccard(intersection, union, eps=1e-15):
    return (intersection) / (union - intersection + eps)

def dice(intersection, union, eps=1e-15, smooth=1.):
    return (2. * intersection + smooth) / (union + smooth + eps)

class BCESoftJaccardDice:

    def __init__(self, bce_weight=0.5, mode="dice", eps=1e-7, weight=None, smooth=1.):
        self.nll_loss = torch.nn.BCEWithLogitsLoss(weight=weight)
        self.bce_weight = bce_weight
        self.eps = eps
        self.mode = mode
        self.smooth = smooth

    def __call__(self, outputs, targets):    
        loss = self.bce_weight * self.nll_loss(outputs, targets)

        if self.bce_weight < 1.:
            targets = (targets == 1).float()
            outputs = torch.sigmoid(outputs)
            intersection = (outputs * targets).sum()
            union = outputs.sum() + targets.sum()
            if self.mode == "dice":
                score = dice(intersection, union, self.eps, self.smooth)
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

class Trainer:
    
    def __init__(self, path=None, gpu=-1, **kwargs):
        
        if path is not None:
            kwargs = pickle.load(open(path+"/model_params.pickle.dat", "rb"))
            kwargs["device_idx"] = gpu
            kwargs["pretrained"], kwargs["reset"] = False, False
            self.path = path
        else:
            self.directory = kwargs["directory"]
            self.path = os.path.join(self.directory, self.model_name)

        self.model_name = kwargs["model_name"]
        self.model_type = kwargs["model"].lower()        
        self.device_idx = kwargs["device_idx"]
        self.cpu = True if self.device_idx < 0 else False
        self.ADAM = kwargs["ADAM"]
        self.pretrained = kwargs["pretrained"]
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        
        self.cp_counter_loss, self.cp_counter_metric = 0, 0
        self.max_lr = .5
        
        net_init_params = {k:v for k, v in kwargs.items() 
            if k in ["Dropout", "pretrained", "num_classes", "num_filters"]
        }

        if self.model_type == "mobilenetv2":
            self.initial_model = UnetMobilenetV2(**net_init_params)
        else:
            net_init_params["model"] = self.model_type
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
                
    def dfs_freeze(self, model):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False if self.freeze_encoder else True
            self.dfs_freeze(child)
        
    def get_model(self, model):
        model = model.train()
        if self.cpu:
            return model.cpu()
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
        for image, mask, mask_w in tqdm(data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers=0)):
            image = image.type(torch.FloatTensor).cuda(self.device_idx)

            it += 1
            current_lr = learning_rate * (lr_mult**it)

            y_pred = self.model(Variable(image))
            if self.model_type == "mobilenetv2":
                y_pred = nn.functional.interpolate(y_pred, scale_factor=2, mode='bilinear', align_corners=True)

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
        self.freeze_encoder = kwargs["freeze_encoder"]

        torch.cuda.empty_cache()
        self.model = self.get_model(self.initial_model)
        
        if self.pretrained and self.freeze_encoder and self.model_type != "mobilenetv2":
            self.dfs_freeze(self.model.conv1)
            self.dfs_freeze(self.model.conv2)
            self.dfs_freeze(self.model.conv3)
            self.dfs_freeze(self.model.conv4)
            self.dfs_freeze(self.model.conv5)
        
        if self.ADAM:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                              lr=learning_rate)
        else:
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                             lr=learning_rate, momentum=0.9, nesterov=True)

        max_lr = self.max_lr
        iterations = len(dataset) // batch_size
        if abs(CLR) == 1:
            iterations *= epoch
        lr_mult = (max_lr/learning_rate)**(1/iterations)
        current_rate = learning_rate
        
        checkpoint_metric, checkpoint_loss, it, k, cooldown = -np.inf, np.inf, 0, 1, 0
        self.history = {"loss":{"train":[], "test":[]}, "metric":{"train":[], "test":[]}}
        
        for e in range(epoch):
            torch.cuda.empty_cache()
            self.model.train()
                    
            if e >= 2 and self.freeze_encoder and self.model_type != "mobilenetv2":
                self.freeze_encoder = False
                self.dfs_freeze(self.model.conv1)
                self.dfs_freeze(self.model.conv2)
                self.dfs_freeze(self.model.conv3)
                self.dfs_freeze(self.model.conv4)
                self.dfs_freeze(self.model.conv5)
                
                if self.ADAM:
                    self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                                      lr=current_rate)
                else:
                    self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                                     lr=current_rate, momentum=0.9, nesterov=True)
                
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
            for image, mask, mask_w in tqdm(data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers=0)):
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
                                self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                                                 lr=current_rate, momentum=0.9, nesterov=True)

                y_pred = self.model(Variable(image))
                if self.model_type == "mobilenetv2":
                    y_pred = nn.functional.interpolate(y_pred, scale_factor=2, mode='bilinear', align_corners=True)

                loss_fn = BCESoftJaccardDice(bce_weight=bce_loss_weight, 
                                             weight=mask_w.cuda(self.device_idx), mode="dice")
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
            self.model.eval()
            val_loss, val_metric = [], []
            for image, mask, mask_w in data.DataLoader(dataset_val, batch_size = batch_size // 2, shuffle = False, num_workers=0):
                image = image.cuda(self.device_idx)

                y_pred = self.model(Variable(image))
                if self.model_type == "mobilenetv2":
                    y_pred = nn.functional.interpolate(y_pred, scale_factor=2, mode='bilinear', align_corners=True)
                
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
        plt.savefig(self.path + '/%s_history.png' % mode)
        
    def load_state(self, path=None, mode="metric", load_optimizer=True):
        if load_optimizer: load_optimizer = self.optimizer
        if path is None:            
            path = self.path+'/%s_checkpoint_%s.pth' % (self.model_name, mode)
        load_checkpoint(path, self.model, load_optimizer, self.cpu)

    def predict_mask(self, imgs, biggest_side=None, denoise_borders=False):
        if not self.cpu:
            torch.cuda.empty_cache()
        if imgs.ndim < 4:
            imgs = np.expand_dims(imgs, axis=0)
        l, h, w, c = imgs.shape
        w_n, h_n = w, h
        if biggest_side is not None: 
            w_n = int(w/h * min(biggest_side, h))
            h_n = min(biggest_side, h)
        
        wd, hd = w_n % 32, h_n % 32
        if wd != 0: w_n += 32 - wd
        if hd != 0: h_n += 32 - hd
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        all_predictions = []
        for i in range(imgs.shape[0]):
            img = self.norm(cv2.resize(imgs[i], (w_n, h_n), interpolation=cv2.INTER_LANCZOS4))
            img = img.unsqueeze_(0)
            if not self.cpu:
                img = img.type(torch.FloatTensor).cuda(self.device_idx)
            else:
                img = img.type(torch.FloatTensor)
            output = self.model(Variable(img))
            if self.model_type == "mobilenetv2":
                output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
            output = torch.sigmoid(output)
            output = output.cpu().data.numpy()
            y_pred = np.squeeze(output[0])
            y_pred = remove_small_holes(remove_small_objects(y_pred > .3))
            y_pred = (y_pred * 255).astype(np.uint8)
            y_pred = cv2.resize(y_pred, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
            _,alpha = cv2.threshold(y_pred.astype(np.uint8),0,255,cv2.THRESH_BINARY)
            b, g, r = cv2.split(imgs[i])
            bgra = [r,g,b, alpha]
            y_pred = cv2.merge(bgra,4)
            if denoise_borders:
                #denoise mask borders
                y_pred[:, :, -1] = cv2.morphologyEx(y_pred[:, :, -1], cv2.MORPH_OPEN, kernel)
            all_predictions.append(y_pred)
        return all_predictions

def split_video(filename, frame_rate=12):
    vidcap = cv2.VideoCapture(filename)
    frames = []
    succ, frame = vidcap.read()
    h, w = frame.shape[:2]
    center = (w / 2, h / 2)
    while succ:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(frame[:, ::-1, :], axes=[1,0,2])
        frames.append(frame)
        succ, frame = vidcap.read()
    return np.array(frames).astype(np.uint8)[::24 // frame_rate]

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

def n_unique_permuts(n, r):
    return factorial(n) / (factorial(r)*factorial(n-r))

def save_images(out, path="./data/gif_test"):
    letters = string.ascii_lowercase
    r = 0; n_uniques = 0
    while n_uniques < len(out):
        r += 1
        n_uniques = n_unique_permuts(len(letters), r)
    names = list(itertools.combinations(letters, r))
    for im, fname in zip(out, names[:len(out)]):
        cv2.imwrite(path+"/%s.png" % ("".join(fname)), im)