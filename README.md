# Using computer vision for positioning 

Most smartphones can use GPS or Wifi for estimating their location, here an alternative method of using 
computer vision for indoor & outdoor positioning is studied. 

--- 

### 01. Indoor CV ### 

First, a location is mapped beforehand by taking photos of distinctive features in the environment and by capturing 3D depth data using an Intel RealSense camera. The features found in the photos are then tagged with their 3D world coordinates. 

<img src="https://user-images.githubusercontent.com/13679090/68001324-949f2200-fc9e-11e9-8e00-c160ec93465b.png" width="720"> 
<img src="https://user-images.githubusercontent.com/13679090/68001325-949f2200-fc9e-11e9-966f-862958a0b334.png" width="720">
<i>Above: sample photos taken at a playground with corresponding depth data on the right.</i><br/><br/><br/>  

Second, on the smartphone, feature-matching is applied using the AKaze algorithm (since SIFT is patented) between the reference photos taken above and the live camera feed. Then, pose-estimation is applied by feeding the matched 3D world coordinates from earlier. 

With that, the smartphone can estimate its distance to the relevant environmental features, and hence estimate its location. 

<img src="https://user-images.githubusercontent.com/13679090/68001326-949f2200-fc9e-11e9-9f54-e6c56448231a.png" width="640"> 
<img src="https://user-images.githubusercontent.com/13679090/68001327-9537b880-fc9e-11e9-8db3-9d016cfda939.png" width="640">
<i>Above: screenshots on iPhone app showing features matched to the reference photos & distances to the objects estimated.</i><br/><br/>  

<hr/> 

<strong>Limitations</strong>  

- the biggest drawback is that the features identified in the smartphone feed could be matched to the wrong features in the reference photos, causing the pose to be computed from the wrong world coordinates and in turn causing poor location accuracy. 
- one alternative would be to match homographies instead of points individually, however this constrains the features mappable to only flat surfaces. 
- another alternative would be to match patterns connecting the points instead of points individually, however at this time an algorithm for generating matchable patterns of points isn't known to be available yet. 

Overall, while there might be promising improvements to this localization method, the emergence of newer smartphones with multiple back cameras that could be used for live depth mapping (& thus SLAM) as well as better augmented reality features shipped with mobile OSes could eventually prove to be more useful for visual positioning, especially for visually-impaired users.  

 <br/>Useful Links: 

Intel RealSense https://github.com/IntelRealSense/librealsense/wiki <br/>
OpenCV with iOS https://medium.com/pharos-production/using-opencv-in-a-swift-project-679868e1b798 <br/>
OpenCV with iOS https://medium.com/@dwayneforde/image-recognition-on-ios-with-swift-and-opencv-b5cf0667b79 <br/>
OpenCV with iOS https://medium.com/@yiweini/opencv-with-swift-step-by-step-c3cc1d1ee5f1 

--- 

### 02. Benchmarks ### 

See [BENCHMARK.md](https://github.com/SkinnyRat/CV-Stuff/blob/master/BENCHMARK.md). Convolutional neural network training with TensorFlow and Caffe2 for MNIST and CIFAR10 on ImageNet (yep I know, not the best idea). 

Speed & accuracy benchmarked for GTX 1050, GTX 1060, GTX 1080Ti, and 2x GTX 1080Ti GPUs. This section is not directly related to the visual positioning technique described above. 
