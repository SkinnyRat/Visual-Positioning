import os, sys, math, keyboard 
import numpy as np 
from PIL import Image 
from ftplib import FTP_TLS 
from matplotlib import pyplot as plt 


DAT = "Data/Directory/" 
FN  = "" 
if len(sys.argv) == 2: FN = sys.argv[1] 

with open(DAT+"Point3f.csv", "w") as Out: Out.write('') 
#with open(DAT+"Point2f.csv", "w") as Out: Out.write('') 

DIR   = "Data/Directory/" 
rows  = [line.rstrip('\n') for line in open(DIR+"RGBD"+FN+".dat")] 
photo = plt.imread(DIR+"Color"+FN+".png") 

tap = 1 
wc1 = ['0','0','0'] 
wc2 = ['0','0','0'] 
fig, ax = plt.subplots() 


def Formater(x, y): 
	try: 
		cols = rows[int(y)].split(' ') 
		if cols[int(x)] == "0.00,0.00,0.00": 
			return "World: NULL!\t\t" 
		return "World: " + cols[int(x)] + "\t\t" 
	except: 
		return "World: NULL!\t\t" 


def Click(event): 
	global plt; global tap; global wc1; global wc2 
	cols   = rows[int(event.ydata)].split(' ') 
	coords = cols[int(event.xdata)].split(',') 
	
	#with open(DAT+"Point3f.csv", "a") as Out: 
	#	Out.write(coords[0]+' '+coords[1]+' '+coords[2]+'\n') 
	#with open(DAT+"Point2f.csv", "a") as Out: 
	#	Out.write(str(int(event.xdata))+" "+str(int(event.ydata))+'\n') 
	
	if tap == 1: 
		wc1 = [float(coords[0]), float(coords[1]), float(coords[2])] 
		tap = 2 
	else: 
		wc2 = [float(coords[0]), float(coords[1]), float(coords[2])] 
		tap = 1 
		Lng = (abs(wc1[0]-wc2[0]))**2 + (abs(wc1[1]-wc2[1]))**2 + (abs(wc1[2]-wc2[2]))**2 
		print("(",wc1[0],",",wc1[1],",",wc1[2],") to (",wc2[0],",",wc2[1],",",wc2[2],")") 
		plt.title('Length = %.2f cm'%(math.sqrt(Lng) * 100)) 
		plt.gcf().canvas.draw_idle() 


ttl  = plt.title('Frame '+str(FN)) 
im   = ax.imshow(photo, interpolation='none') 
ax.format_coord = Formater 
fig.canvas.mpl_connect('button_press_event', Click) 
fig.tight_layout() 
plt.show() 

print("Save data to cloud?\n") 
while True: 
	if keyboard.is_pressed('\r'): 
		OriginalPNG  = Image.open(DIR+"Color"+FN+".png")
		ConvertedJPG = OriginalPNG.convert('RGB'); 
		ConvertedJPG.save(DIR+"Color"+FN+".jpg") 
		
		ftp = FTP_TLS('FTP server goes gere') 
		ftp.login('Username', 'Password') 
		ftp.cwd("site/wwwroot/") 
		ftp.storlines('STOR '  + "DepthMapData.dat",   open(DIR+"RGBD"+FN+".dat",  'rb')) 
		ftp.storbinary('STOR ' + "ReferenceFrame.jpg", open(DIR+"Color"+FN+".jpg", 'rb')) 
		ftp.quit() 
		
		os.remove(DIR+"Color"+FN+".jpg") 
		print("Data from Frame "+str(FN)+" saved to cloud.\n") 
		break 
	elif keyboard.is_pressed(' '): 
		print("Exited without uploading to cloud.\n") 
		break 
	else: pass 

# Press Enter to upload to cloud, Space to exit. 

