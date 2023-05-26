
import os

root_dir  = './Segmentation/data/Generated_Dataset_Test/h5/'
note =''
for h5_name in os.listdir(root_dir):
	# note = note+'./'+root_dir+h5_name+'\n'
	note = note + root_dir+h5_name+'\n'

f = open('./Segmentation/data/Generated_Dataset_Test_train.txt','w')
f.write(note)
f.close()
