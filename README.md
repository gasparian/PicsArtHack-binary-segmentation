# PicsArtHack-binary-segmentation

The goal of the hackathon was to build some image processing alogrithm which can be helpful for [PicsArt](https://picsart.com/?hl=en) applications.  
Here I publish results of the first stage: segmenting people on selfies.
PicsArt gives us labeled [dataset](https://drive.google.com/file/d/1_e2DcZnjufx35uSmQElN5mpdo-Rlv7ZI/view?usp=sharing). [Dice](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) coef. was used as evaluation metric.  
I noticed that a lot of images has been labeled by another segmentation model due to a lot of artifacts around the masks borders. Also in test dataset apperas copies of train set images. So after training, I did not expect good results on images "from the wild".

### 1. Loss  
For this problem I used fairly common bce-dice loss. So the algorithm is simple: we take a logits output from model and put it inside binary cross-enthropy loss and the natural logarithm of dice loss (after passing sigmoid function). After that we only need to combine these losses with weights:
```
dice_loss = (2. * intersection + eps) / (union + eps)
loss = w * BCELoss + (1 - w) * log(dice_loss) * (-1)
```  
Also, in this case, we don't need to tune tresholds of final pseudo-probabilities (after sigmoid).  
Finally we can adjust weights to the mask (I did it inside BCELoss), to penalize model for mistakes around the mask borders. For this purpose we can use opencv erosion kernel-operation:
```
def get_mask_weight(mask):
    mask_ = cv2.erode(mask, kernel=np.ones((8,8),np.uint8), iterations=1)
    mask_ = mask-mask_
    return mask_ + 1
```  
On the picture below we can see how the input data looks like:    
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/example_1.png">  
### 2. Training  
I used modification of **unet** (which is well recommended for solving binary semantic segmentation problems) with two encoders pretrained on Imagenet: resnet101 and mobilenetV2. One of the goals was to compare the performance of "heavy" and "light" encoders.  
You can check all training params inside `train.py`.

```
python3 train.py --train_path ./data/train_data --workdir ./data/  --model_type mobilenetV2
```

Data augmentation was provided via brilliant [albumentaions](https://github.com/albu/albumentations) library.  
Inside the `utils.py` code you can find learning rate scheduling, encoder weights freezeing and some other useful hacks which can help to train networks in more efficient way. Also passing the parameter `model_type` you are able to choose one of the predefined models based on: resnet18, resnet34, resnet50, resnet101, mobilenetV2. 

So, in the end I've got two trained models with close Dice values on a validation set. Here is a few numbers:    

Encoder: | ResNet101             |  MobileNetV2  
:-------------------------:|:-------------------------:|:-------------------------:  
epochs (best of 200) | 177 | 173  
Dice | 0.987 (0.988) | 0.986 (0.988)  
loss | 0.029 (0.022) | 0.030 (0.024)  
No. of parameters | 120 131 745 | 4 682 912  

ResNet101 evaluation process:  
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/resnet101_loss.png">  <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/resnet101_metric.png">  
MobileNetV2 evaluation process:  
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/mobilenetV2_loss.png">  <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/mobilenetV2_metric.png">  

I want to point that despite the fact that mobilenetV2 has ~x26 less weights and at the same time we are able to get models with pretty similar quality, we did it **with this particullar problem using mentioned dataset**. So I don't recommend to extend these results to other classification problems.

### 3. Tests  
Inference time comparison with input images 320x256 from the test-set:  

Device | ResNet101 | MobileNetV2  
:-------------------------:|:-------------------------:|:-------------------------:  
AMD Threadripper 1900X CPU (1 thread) | 2.19 s ± 10.2 ms | 439 ms ± 4.39 ms  
GTX 1080Ti GPU | 44.9 ms ± 2 ms | 28.8 ms ± 2.86 ms  

Often, output masks contain some noise on the borders (which is become more annoying on large images), so we can try to fix it applying morhological transform: 
```
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
y_pred[:, :, -1] = cv2.morphologyEx(y_pred[:, :, -1], cv2.MORPH_OPEN, kernel)
```  
Original | Trnasformed  
:-------------------------:|:-------------------------:
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/ex_3_orig_mask.png"> | <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/ex_3_edited_mask.png">  

Additionaly we can transform segmented images. For instance let's make a gaussian blur of a background:
```
blurred = cv2.GaussianBlur(test_dataset[n],(21,21),0)
dst = cv2.bitwise_and(blurred, blurred, mask=~out[0][:, :, -1])
dst = cv2.add(cv2.bitwise_and(test_dataset[n], test_dataset[n], mask=out[0][:, :, -1]), dst)
```
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/ex_2_orig.png">  <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/ex_2_transformed.png">  <img src="https://github.com/gasparian/PicsArtHack-binary-segmentation/blob/master/pics/girl_ex_orig.png">  <img src="https://github.com/gasparian/PicsArtHack-binary-segmentation/blob/master/pics/girl_ex_blured.png">  

And actually we can process videos too (see `predict.py`). Example below is a video made by me with a cellphone (original image size: 800x450):  

<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/VID_orig.gif" height=384>  <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/VID_edited.gif" height=384>  

These results has been obtained with mobilenetV2 model. You can play with it too, here is it's [weights](https://drive.google.com/file/d/1XSRaOaoWKKSllIuUgkW0BVsMKieQ8mbG/view?usp=sharing).  

```
python3 predict.py -p ./test --model_path ./models/mobilenetV2_model --gpu -1 --frame_rate 12 --denoise_borders --biggest_side 320
```
This script reads all the data inside `-p` folder: both pictures and videos.

### 4. Porting model to IOS device  
Finally, we can convert trained mobilenetV2 model with CoreML to make inference on the IOS devices. To make this happen, don't keep encoder layers separatly inside the model class - use them in forward pass. Also, with the certain versions of torch, onnx and coreml (see `requirements.txt`), you can't convert upsampling / interpolation layers. Hope it will be fixed in the future releases.

```
python3 CoreML_convert.py --tmp_onnx ./models/tmp.onnx  --weights_path ./models/mobilenetV2_model/mobilenetV2_model_checkpoint_metric.pth
```

### 5. Environment  
For your own experiments I highly recommend to use [Deepo](https://github.com/ufoym/deepo) as a fast way to deploy universal deep-learning environment inside a Docker container.  Other dependencies can be found in `requirements.txt`.  
