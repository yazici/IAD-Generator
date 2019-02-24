import os
from os.path import join, isfile, isdir

import sys
import rosbag

import numpy as np

def read_files_in_dir(directory):
	contents = [join(directory, f) for f in os.listdir(directory)]

	all_files = []
	for f in contents:
		if isfile(f) and f[-4:] == ".bag":
			all_files += [f]
		elif isdir(f):
			all_files += read_files_in_dir(f)
	
	return all_files

filenames = read_files_in_dir(sys.argv[1])

lengths = []

for f in filenames:
	print(f)
	bag = rosbag.Bag(f)

	count  = 0
	for x,y,z in bag.read_messages(topics=["/kinect2/rgb/image/compressed"]):
		count +=1

	file_name_sep = f.split('/')
	label_name = file_name_sep[-1].split('_')[0]
	actual_label = file_name_sep[-2]

	new_filename = actual_label+file_name_sep[-1][len(label_name):]

	if(label_name!= actual_label):
		file_prefix = ""

		for i in file_name_sep[:-1]:
			file_prefix = join(file_prefix, i)

		cmd = "mv "+f+' '+file_prefix+'/'+new_filename
		print(cmd)
		os.system(cmd)

	lengths.append( count )
	
lengths = np.array(lengths)
print("longest length is: ", np.max(lengths))