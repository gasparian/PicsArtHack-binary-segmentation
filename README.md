# PicsArt-Hack-binary_segmentation

The goal of the hackathon was to build a solution for image processing which can be helpful for [PicsArt](https://picsart.com/?hl=en) applications.  
Here I publish results for the first stage: segmenting people on photos.
PicsArt gives us labeled [dataset](https://drive.google.com/file/d/1_e2DcZnjufx35uSmQElN5mpdo-Rlv7ZI/view?usp=sharing).  
[Dice](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) coef. was used as evaluation metric.  

[MobileNetV2 model](https://drive.google.com/file/d/1mMtNNPRvc7DVC-Ozu2ne5cXaOrVNY7Dm/view?usp=sharing)  

<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/example_1.png">  

Epoch: 34, Train loss: 0.033, Train metric: 0.984, Val loss: 0.040, Val metric: 0.982  
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/resnet101_loss.png">  <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/resnet101_metric.png">  

Epoch: 193, Train loss: 0.038, Train metric: 0.982, Val loss: 0.047, Val metric: 0.980  
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/mbv2_loss.png">  <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/mbv2_metric.png">  

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
