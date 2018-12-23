# PicsArt-Hack-binary_segmentation

The goal of the hackathon was to build some image processing alogrithm which can be helpful for [PicsArt](https://picsart.com/?hl=en) applications.  
Here I publish results of the first stage: segmenting people on photos.
PicsArt gives us labeled [dataset](https://drive.google.com/file/d/1_e2DcZnjufx35uSmQElN5mpdo-Rlv7ZI/view?usp=sharing).  
[Dice](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) coef. was used as evaluation metric.  

### 1. Loss.  
For this problem I used fairly common bce-dice loss. So the algorithm is simple: we take a logits output from model and put it inside binary cross-enthropy loss and the natural logarithm of dice loss (after passing sigmoid function). After that we only need to combine these losses with weights:
```
dice_loss = (2. * intersection + eps) / (union + eps)
loss = w * BCELoss + (1 - w) * log(dice_loss) * (-1)
```  
Also, after applying this loss, we don't need to tune tresholds of final pseudo-probabilities (after sigmoid).  
Finally we can adjust weight on mask (I do it inside BCELoss), to penalize model for mistakes around the mask borders. For this purpose we can use opencv window-operation called `erode`:
```
def get_mask_weight(mask):
    mask_ = cv2.erode(mask, kernel=np.ones((8,8),np.uint8), iterations=1)
    mask_ = mask-mask_
    return mask_ + 1
```  
On the picture below we can see how input data looks like:    
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/example_1.png">  

Epoch: 34, Train loss: 0.033, Train metric: 0.984, Val loss: 0.040, Val metric: 0.982  
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/resnet101_loss.png">  <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/resnet101_metric.png">  

Epoch: 193, Train loss: 0.038, Train metric: 0.982, Val loss: 0.047, Val metric: 0.980  
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/mbv2_loss.png">  <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/mbv2_metric.png">  

```
blurred = cv2.GaussianBlur(test_dataset[n],(21,21),0)
dst = cv2.bitwise_and(blurred, blurred, mask=~out[0][:, :, -1])
dst = cv2.add(cv2.bitwise_and(test_dataset[n], test_dataset[n], mask=out[0][:, :, -1]), dst)
```
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/ex_2_orig.png">  <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/ex_2_transformed.png">  

Original             |  Segmented
:-------------------------:|:-------------------------:
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/VID_orig.gif" height=384>  |  <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/VID_edited.gif" height=384>      

resnet101: 44,549,160
mobilenetV2: 6,906,767

photos 320x256

437 ms cpu mobilenetv2
2140 ms cpu resnet101
~5x slower

24 ms gpu mobilenetv2
43 ms gpu resnet101
~2x slower
[MobileNetV2 model](https://drive.google.com/file/d/1mMtNNPRvc7DVC-Ozu2ne5cXaOrVNY7Dm/view?usp=sharing)  
