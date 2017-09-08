import os, shutil, cv2 
import numpy as np 
import caffe2.python.predictor.predictor_exporter as pe 

from matplotlib    import pyplot 
from caffe2.proto  import caffe2_pb2 
from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew 

core.GlobalInit(['caffe2', '--caffe2_log_level=0']) 


current_folder = os.path.join(os.path.expanduser('~'), 'Documents/DeepLearning/MNIST/Caffe2/Data')
data_folder    = os.path.join(current_folder, 'tutorial_data',  'mnist')
root_folder    = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')

png = np.zeros((100, 1, 28, 28)) 
for n in range(0, 10): 
    for i in range(0, 10): 
        png[n*10+i]  = (cv2.imread("Plots/M"+str(i+1)+".png", cv2.IMREAD_GRAYSCALE)) 
        png[n*10+i]  = png[n*10+i].astype(np.float32) 
        png[n*10+i] /= 255.0 
png  =  png.astype(np.float32) 


predict_net = pe.prepare_prediction_net(os.path.join(root_folder, "mnist_model.minidb"), "minidb")
print("Workspace blobs loaded : {}".format(workspace.Blobs()))

workspace.FeedBlob("data",   png)
workspace.RunNetOnce(predict_net)

pyplot.figure()
_       = visualize.NCHW.ShowMultiple(png)
pyplot.figure()
softmax = workspace.FetchBlob('softmax')
_       = pyplot.plot(softmax[9],  'ro')
pyplot.title('Prediction for selected image')
pyplot.show()



