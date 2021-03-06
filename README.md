# TensorFlowExamples

## 1. SRCNN (Super resolution)

SRCNN (http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) implementation using Tensorflow

Dataset Stanford BG (http://dags.stanford.edu/data/iccv09Data.tar.gz)

![Image](https://github.com/gnoses/TensorFlowExamples/blob/master/Images/srcnn.png)


## 2. Convolutional Autoencoder

Down sampling (conovolution stride 2), Up sampling (transpose convolution with stride 2)


## 3. Convolutional Encoder Decoder Net

Encoder decoder net for Semantic segmentation using pooling and unpooling

![Image](https://github.com/gnoses/TensorFlowExamples/blob/master/Images/SemanticSegmentation.png)

Instruction :

Download dataset and unpack into src folder (CamVid https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid from Cambridge University Machine Intelligence Lab)

Run CreateDB script once in main code

    CreateDB('CamVid/train')
    CreateDB('CamVid/val')
    
Training plot

![Image](https://github.com/gnoses/TensorFlowExamples/blob/master/Images/CED_training_plot.png)


## 4. SVHN Classification 

![Image](http://ufldl.stanford.edu/housenumbers/32x32eg.png)

Dataset : http://ufldl.stanford.edu/housenumbers/

