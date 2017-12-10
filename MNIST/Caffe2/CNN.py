import os, shutil, time 
import numpy as np 
import caffe2.python.predictor.predictor_exporter as pe 

from matplotlib    import pyplot 
from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew 

core.GlobalInit(['caffe2', '--caffe2_log_level=0']) # Details, --caffe2_log_level=-1


# This section preps training and test set in lmdb database

def DownloadResource(url, path):
    print("Downloading... {} to {}".format(url, path))
    import requests, zipfile, StringIO
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall(path)
    print("Completed download and extraction.")

current_folder = os.path.join(os.path.expanduser('~'), 'Documents/DeepLearning/MNIST/Caffe2/Data')
data_folder    = os.path.join(current_folder, 'tutorial_data',  'mnist')
root_folder    = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')
db_missing     = False
db_url         = "http://download.caffe2.ai/databases/mnist-lmdb.zip"

if not os.path.exists(data_folder):
    os.makedirs(data_folder)   
    print("Data folder not found! Folder created: {}".format(data_folder))

if os.path.exists(os.path.join(data_folder, "mnist-train-nchw-lmdb")):
    print("Training  database found.")
else:
    db_missing = True
if os.path.exists(os.path.join(data_folder,  "mnist-test-nchw-lmdb")):
    print("Testing   database found.")
else:
    db_missing = True

if db_missing:
    print("Training and/or testing databases not found.")
    try:
        DownloadResource(db_url, data_folder)
    except Exception as ex:
        print("Failed to download dataset. Please download it manually from {}".format(db_url))
        print("Unzip it and place the two database folders here: {}".format(data_folder))
        raise ex

if os.path.exists(root_folder):
    print("Previous  files found, cleaning up...")
    shutil.rmtree(root_folder)

os.makedirs(root_folder)
workspace.ResetWorkspace(root_folder)
print("Workspace root:" + root_folder)


def AddInput(model, batch_size, db, db_type):
    data_uint8, label = model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size=batch_size, db=db, db_type=db_type)
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    data = model.Scale(data, data, scale=float(1./256))
    data = model.StopGradient(data, data)
    return data, label

'''
This part is the standard LeNet model: from data to the softmax prediction.
For each convolutional layer we specify dim_in - number of input channels
and dim_out - number or output channels. Also each Conv and MaxPool layer changes the
image size. For example, kernel of size 5 reduces each side of an image by 4.
While when we have kernel and stride sizes equal 2 in a MaxPool layer, it divides
each side in half.
'''

def AddLeNetModel(model, data):
    # Image size: 28 x 28 -> 24 x 24
    # Image size: 24 x 24 -> 12 x 12
    # Image size: 12 x 12 ->  8 x  8
    # Image size: 8 x 8   ->  4 x  4
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by image size
    conv1    = brew.conv(model,     data,  'conv1', dim_in=1,  dim_out=20,  kernel=5)
    pool1    = brew.max_pool(model, conv1, 'pool1', kernel=2,  stride =2)
    conv2    = brew.conv(model,     pool1, 'conv2', dim_in=20, dim_out=100, kernel=5)
    pool2    = brew.max_pool(model, conv2, 'pool2', kernel=2,  stride =2)
    fc3      = brew.fc(model,       pool2, 'fc3',   dim_in=100 *  4 *  4,   dim_out=500)
    relu     = brew.relu(model,     fc3,    fc3)
    pred     = brew.fc(model,       relu,  'pred',  500, 10)
    softmax  = brew.softmax(model,  pred,  'softmax')
    return   softmax

def AddAccuracy(model, softmax, label):
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return   accuracy

def AddTrainingOperators(model,     softmax, label):
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    loss = model.AveragedLoss(xent, "loss")
    AddAccuracy(model, softmax, label)
    model.AddGradientOperators([loss])
    ITER = brew.iter(model,    "iter")
    LR   = model.LearningRate(ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999)
    ONE  = model.param_init_net.ConstantFill([], "ONE", shape=[1],     value=1.0)
    for param in model.params:
        param_grad = model.param_to_grad[param]
        model.WeightedSum([param, ONE, param_grad, LR], param)

def AddBookkeepingOperators(model):
    model.Print('accuracy', [], to_file=1)
    model.Print('loss',     [], to_file=1)
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)

arg_scope    = {"order": "NCHW"}
train_model  = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope)
data, label  = AddInput(train_model, batch_size=100, db=os.path.join(data_folder, 'mnist-train-nchw-lmdb'), db_type='lmdb')
softmax      = AddLeNetModel(train_model, data)

train_model.param_init_net.RunAllOnGPU()
train_model.net.RunAllOnGPU()

AddTrainingOperators(train_model, softmax, label)
AddBookkeepingOperators(train_model)


'''
Testing model. We will set the batch size to 100, so that the testing
pass is 100 iterations (10,000 images in total).
For the testing model, we need the data input part, the main LeNetModel
part, and an accuracy part. Note that init_params is set False because
we will be using the parameters obtained from the train model.
'''

test_model   = model_helper.ModelHelper(name="mnist_test", arg_scope=arg_scope, init_params=False)
data, label  = AddInput(test_model, batch_size=100, db=os.path.join(data_folder, 'mnist-test-nchw-lmdb'), db_type='lmdb')
softmax      = AddLeNetModel(test_model, data)
deploy_model = model_helper.ModelHelper(name="mnist_deploy", arg_scope=arg_scope, init_params=False)

test_model.param_init_net.RunAllOnGPU()
test_model.net.RunAllOnGPU()

AddAccuracy(test_model, softmax, label)
AddLeNetModel(deploy_model, "data")


with open(os.path.join(root_folder, "train_net.pbtxt"),      'w') as fid:
    fid.write(str(train_model.net.Proto()))
with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "test_net.pbtxt"),       'w') as fid:
    fid.write(str(test_model.net.Proto()))
with open(os.path.join(root_folder, "test_init_net.pbtxt"),  'w') as fid:
    fid.write(str(test_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "deploy_net.pbtxt"),     'w') as fid:
    fid.write(str(deploy_model.net.Proto()))
print("Protocol  buffers created in workspace root.")

START       = time.time()
total_iters = 20000
accuracy    = np.zeros(total_iters)
loss        = np.zeros(total_iters)

workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net, overwrite=True)

for i in range(total_iters):
    workspace.RunNet(train_model.net)
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i]     = workspace.FetchBlob('loss')

pyplot.plot(loss, 'b')
pyplot.plot(accuracy, 'r')
pyplot.legend(('Loss', 'Accuracy'), loc='upper right')
#pyplot.show()
pyplot.figure()
data    = workspace.FetchBlob('data')
_       = visualize.NCHW.ShowMultiple(data)
pyplot.figure()
softmax = workspace.FetchBlob('softmax')
_       = pyplot.plot(softmax[0], 'ro')
pyplot.title('Prediction for the first image')
#pyplot.show()

STOP  = time.time()
print "\nTRAINING  COMPLETE" 
print "TIME   = ", (STOP-START), "s.\n" 


'''
# Convolutions for this mini-batch
pyplot.figure()
conv     = workspace.FetchBlob('conv1')
shape    = list(conv.shape)
shape[1] = 1
conv     = conv[:,15,:,:].reshape(shape)
_        = visualize.NCHW.ShowMultiple(conv)
'''

test_iters    = 200
test_accuracy = np.zeros(test_iters)

workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net, overwrite=True)

for i in range(test_iters):
    workspace.RunNet(test_model.net.Proto().name)
    test_accuracy[i] = workspace.FetchBlob('accuracy')
print('Test  Accuracy: ' + str(round(test_accuracy.mean()*100, 3)) + '%')

pyplot.plot(test_accuracy, 'r')
pyplot.title('Acuracy over test batches.')
#pyplot.show()

pe_meta = pe.PredictorExportMeta(predict_net=deploy_model.net.Proto(), parameters=[str(b) for b in deploy_model.params], inputs=["data"], outputs=["softmax"])
pe.save_to_db("minidb", os.path.join(root_folder, "mnist_model.minidb"), pe_meta)
print("Model saved to: " + root_folder + "/mnist_model.minidb")


# Workspace is reset and re-loaded. 

blob = workspace.FetchBlob("data")
workspace.ResetWorkspace(root_folder)
print("Workspace blobs reset  : {}".format(workspace.Blobs()))

predict_net = pe.prepare_prediction_net(os.path.join(root_folder, "mnist_model.minidb"), "minidb")
print("Workspace blobs loaded : {}".format(workspace.Blobs()))

workspace.FeedBlob("data", blob)
workspace.RunNetOnce(predict_net)

pyplot.figure()
_       = visualize.NCHW.ShowMultiple(data)
pyplot.figure()
softmax = workspace.FetchBlob('softmax')
_       = pyplot.plot(softmax[1], 'ro')
pyplot.title('Prediction for selected image')
#pyplot.show()




