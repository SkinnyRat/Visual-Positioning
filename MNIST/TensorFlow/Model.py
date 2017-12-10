import time, os,  urllib, cv2 
import numpy as   np 
import tensorflow as tf 

tf.logging.set_verbosity(tf.logging.ERROR)

sess  = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
saver = tf.train.import_meta_graph('Plots/Saved_Model-2000.meta')
saver.restore(sess, tf.train.latest_checkpoint('Plots/'))

png = np.zeros((100, 1, 28, 28)) 
for n in range(0, 10): 
  for i in range(0, 10): 
    png[n*10+i]  = (cv2.imread("Data/M"+str(i+1)+".png", cv2.IMREAD_GRAYSCALE)) 
    png[n*10+i]  = png[n*10+i].astype(np.float32) 
    png[n*10+i] /= 255.0 
png = png.astype(np.float32) 

graph     = tf.get_default_graph()
x         = graph.get_tensor_by_name("X_:0")
keepprob  = graph.get_tensor_by_name("KEEP:0")
predict   = graph.get_tensor_by_name("Prediction:0")

with sess:
  #for n in tf.get_default_graph().as_graph_def().node:
  #  print n.name 
  
  for n in range(10): 
    pimg   = np.reshape(png[n], (1, 784)) 
    temp   = np.full((1, 10), 1) 
    tpred  = tf.argmax(predict, 1) 
    result = predict.eval(feed_dict={x:pimg, keepprob:1.0}, session=sess) 
    rnge   = np.ptp(result, axis=1) 
    rmin   = np.amin(result) 
    rnorm  = (result - rmin) / rnge 
    print  "Result for M"+str(n+1)+".png =>", rnorm 

print "END OF PROGRAM." 


