# PicsArt-Hack-binary_segmentation

The goal of the hackathon was to build some image processing alogrithm which can be helpful for [PicsArt](https://picsart.com/?hl=en) applications.  
Here I publish results of the first stage: segmenting people on photos.
PicsArt gives us labeled [dataset](https://drive.google.com/file/d/1_e2DcZnjufx35uSmQElN5mpdo-Rlv7ZI/view?usp=sharing).  
[Dice](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) coef. was used as evaluation metric.  

### 1. Loss  
For this problem I used fairly common bce-dice loss. So the algorithm is simple: we take a logits output from model and put it inside binary cross-enthropy loss and the natural logarithm of dice loss (after passing sigmoid function). After that we only need to combine these losses with weights:
```
dice_loss = (2. * intersection + eps) / (union + eps)
loss = w * BCELoss + (1 - w) * log(dice_loss) * (-1)
```  
Also, after applying this loss, we don't need to tune tresholds of final pseudo-probabilities (after sigmoid).  
Finally we can adjust weight on mask (I do it inside BCELoss), to penalize model for mistakes around the mask borders. For this purpose we can use opencv kernel-operation called `erode`:
```
def get_mask_weight(mask):
    mask_ = cv2.erode(mask, kernel=np.ones((8,8),np.uint8), iterations=1)
    mask_ = mask-mask_
    return mask_ + 1
```  
On the picture below we can see how input data looks like:    
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/example_1.png">  
### 2. Training  
I used modification of **unet** (which is well recommended in binary semantic segmentation problems) with two encoders pretrained on Imagenet: resnet101 and mobilenetV2. My goal was to compare the performance of "heavy" and "light" encoders (actually in the case of mobilenet, depthwise-separable convolutions were used in decoder too).  
You can check all training params inside `train.py`, but I want to point a couple things:
 - I freeze pretrained encoder's weights during the first two epochs to tune decoder weights only to decrease convergence time;
 - data augmentation was provided via brilliant [albumentaions](https://github.com/albu/albumentations) library;
 - Inside the `utils.py` code you can find learning rate scheduling, early stopping and some other useful hacks which can help to train networks in more efficient way.  

So in the end I've got two trained models with close metric values on a validation set. Here is a few numbers:    

Characteristic | ResNet101             |  MobileNetV2  
:-------------------------:|:-------------------------:|:-------------------------:  
epochs | 34 | 193  
metric | 0.982 (0.984) | 0.980 (0.982)  
loss | 0.040 (0.033) | 0.047 (0.038)  
No. of parameters | 44 549 160 | 6 906 767  

ResNet101 evaluation process:  
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/resnet101_loss.png">  <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/resnet101_metric.png">  
MobileNetV2 evaluation process:  
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/mbv2_loss.png">  <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/mbv2_metric.png">  

### 3. Tests  
Inference time comparison with input images 320x256 from the test-set:  

Device | ResNet101 | MobileNetV2  
:-------------------------:|:-------------------------:|:-------------------------:  
AMD Threadripper 1900X CPU (single process) | 2140 ms | 437 ms  
GTX 1080Ti GPU | 43 ms | 24 ms  

Additionaly we can transform segmented images, for instance make a gaussian blur of a background:
```
blurred = cv2.GaussianBlur(test_dataset[n],(21,21),0)
dst = cv2.bitwise_and(blurred, blurred, mask=~out[0][:, :, -1])
dst = cv2.add(cv2.bitwise_and(test_dataset[n], test_dataset[n], mask=out[0][:, :, -1]), dst)
```
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/ex_2_orig.png">  <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/ex_2_transformed.png">  

And actually we can process videos too (see `predict.py`). Example below is a video made by me with a cellphone:  

<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/VID_orig.gif" height=384>  <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/VID_edited.gif" height=384>  

These results has been obtained with mobilenetV2 model. You can play with it too, here is it's [weights](https://drive.google.com/file/d/1mMtNNPRvc7DVC-Ozu2ne5cXaOrVNY7Dm/view?usp=sharing).  

### 4. Environment  
For your own experiments I highly recommend to use [Deepo](https://github.com/ufoym/deepo) as a fast way to deploy universal deep-learning environment inside a Docker container.  Other dependencies can be found in `requirements.txt`.  
