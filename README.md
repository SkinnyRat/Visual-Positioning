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

___Machine 2___ <br /> 
CPU: Core i7 7700 <br /> 
RAM: 32 GB DDR4 <br /> 
GPU: 1x GTX 1080 Ti <br /> 
<br /> 

### Benchmark results 
___Caffe 2___ <br /> 
Machine 1 <br /> 
Training: X s <br /> 
Accuracy: X % <br /> 

Machine 2 <br /> 
Training: Y s <br /> 
Accuracy: Y % <br /> 

___TensorFlow___ <br /> 
Machine 1 <br /> 
Training: X s <br /> 
Accuracy: X % <br /> 

Machine 2 <br /> 
Training: Y s <br /> 
Accuracy: Y % <br /> 
<hr />


### URL sources 
1. http://yann.lecun.com/exdb/lenet/ 
2. https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/MNIST.ipynb 
3. https://www.tensorflow.org/get_started/mnist/pros 
4. https://stackoverflow.com/questions/tagged/deep-learning 

In both cases, Ubuntu 16.04 LTS with Cuda 8.0 + cuDNN 5.1 + NCCL 2 + OpenCV 3.1 and gcc 5.3.1 was used. 
<hr /> 


