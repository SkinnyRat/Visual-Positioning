# Benchmark-ImageNet 
Benchmarking tests on the CIFAR-10 model for the 32x32 ImageNet dataset using TensorFlow. <br /> 
Caffe2 was not used here, while CNTK is being considered for future benchmarks. 

### ImageNet and CIFAR10 
The CIFAR-10 model consists of 2x convolution+pooling+normalization layers and 2x fully-connected layers. <br /> 
The ImageNet dataset contains 1.28 million training images in 1000 categories.<br /> 
The ResNet-50/101/152 models are not used for this set of benchmarking tests. 

### Workflow used here 
Training was performed using SGD with 64 images per batch over 1,000,000 iterations per GPU. <br /> 
Training was performed on 1 GPU and then repeated on 2 GPUs. <br /> 
Accuracy was evaluated against 64 images per batch over 1000 iterations on the CPU. <br /> 
Predictions were then tested against single PNG images downloaded from the Web. 
<hr />

### Hardware used here 
___Machine___ <br /> 
CPU: Core i7 7700 <br /> 
RAM: 32 GB DDR4 <br /> 
GPU: 2x GTX 1080 Ti <br /> 
HDD: WD Blue 7200rpm <br /> 
<br /> 

### Benchmark results 
___Single GPU___ <br /> 
Training: ??.?? s <br /> 
Accuracy: ??.?? % <br /> 

___Dual GPUs___ <br /> 
Training: ??.?? s <br /> 
Accuracy: ??.?? % <br /> 
<hr /><hr /><br /> 


# Benchmark-MNIST
Benchmarking tests on the standard LeNet CNN for the MNIST dataset using Caffe2 &amp; TensorFlow. 

### MNIST and LeNet 
The original MNIST data can be downloaded automatically with both Caffe2 and TensorFlow. <br /> 
The LeNet structure used consists of 2 convolution+pooling layers and 1 fully-connected layer. 

### Workflow used here 
Training was performed using SGD with 100 images per batch over 20,000 iterations. <br /> 
Testing accuracy was obtained from 100 images per batch over 200 iterations. <br /> 
Training model was saved to disk and loaded in the separate inference module. <br /> 
Inferences were drawn from 10 png images (that I made) loaded as NumPy arrays. 
<hr />

### Hardware used here 
___Machine 1___ <br /> 
CPU: Pentium G4400 <br /> 
RAM: 8GB DDR4 <br /> 
GPU: 1x GTX 1050 <br /> 
HDD: HGST 5400rpm <br /> 

___Machine 2___ <br /> 
CPU: Core i7 7700 <br /> 
RAM: 32 GB DDR4 <br /> 
GPU: 1x GTX 1080 Ti <br /> 
HDD: WD Blue 7200rpm <br /> 
<br /> 

### Benchmark results 
___Caffe 2___ <br /> 
Machine 1 <br /> 
Training: 267.68 s <br /> 
Accuracy: 98.77 % <br /> 

Machine 2 <br /> 
Training: 91.68 s <br /> 
Accuracy: 98.64 % <br /> 

___TensorFlow___ <br /> 
Machine 1 <br /> 
Training: 282.46 s <br /> 
Accuracy: 99.31 % <br /> 

Machine 2 <br /> 
Training: 106.55 s <br /> 
Accuracy: 99.32 % <br /> 
<hr /><hr /><br/>  

## URL sources 
1. http://image-net.org/download-images <br /> 
2. https://patrykchrabaszcz.github.io/Imagenet32/ <br /> 
3. https://www.cs.toronto.edu/~kriz/cifar.html <br /> 
4. https://www.tensorflow.org/tutorials/deep_cnn <br /> <br /> 
5. http://yann.lecun.com/exdb/lenet/ <br /> 
6. https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/MNIST.ipynb <br /> 
7. https://www.tensorflow.org/get_started/mnist/pros <br /> 
8. https://stackoverflow.com/questions/tagged/deep-learning <br /> 

In both cases, Ubuntu 16.04 LTS with Cuda 8.0 + cuDNN 5.1 + NCCL 2 + OpenCV 3.1 and gcc 5.3.1 was used. 
<hr /> 
