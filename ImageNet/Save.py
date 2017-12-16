import os,  pickle 
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
 
from   PIL import Image 
 
mpl.use('Agg') 
map_file = 'LABELS'
 
 
def unpickle(file): 
    with open(file, 'rb') as fo: 
        dict = pickle.load(fo) 
    return dict
 
def loadData(infile): 
    d = unpickle(infile) 
    x = d['data'] 
    y = d['labels'] 
    #x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:])) 
    #x = x.reshape((x.shape[0],  32, 32, 3)) 
    try: 
        m = d['mean'] 
        m = m.reshape((32, 32, 3)) 
        m = m/np.float32(255) 
    except: 
        m = np.zeros((32,  32, 3)) 
    #x  = x/np.float32(255) 
    #x -= m 
    #x = x.clip(min=0) 
    return x, y, m 
 
 
for n in range(10): 
    x,y,m  = loadData('Training'+str(n+1)) 
    with open('Training'+str(n+1)+'.bin', 'wb') as F: 
        for i in range(128000): 
            L = y[i] 
            I = x[i] 
            F.write(bytearray([int(L/256), L%256])) 
            F.write(bytearray(I)) 
    print("Done with Training"+str(n+1)+" data.") 
 
x,y,m  = loadData('Validation') 
with open('Validation.bin', 'wb') as F: 
    for i in range(50000): 
        L = y[i] 
        I = x[i] 
        F.write(bytearray([int(L/256), L%256])) 
        F.write(bytearray(I)) 
print("Done with Validation data.") 

