from file_io import RosBagFileReader, ITR_NAMES
from iad_gen_global_norm import convert_videos_to_IAD, global_norm

import sys, time, os

from multiprocessing import RawArray, Queue

import tensorflow as tf


def read_files_in_dir(directory):
	contents = [os.path.join(directory, f) for f in os.listdir(directory)]

	all_files = []
	for f in contents:
		if os.path.isfile(f) and f.find(".bag") >=0:
			all_files += [f]
		elif os.path.isdir(f):
			all_files += read_files_in_dir(f)
	
	return all_files

if __name__ == '__main__':

	assert len(sys.argv) >= 2, "Usage: python iad_generator.py <src_directory> -o <out_directory>"

	output_dir = "generated_iads/"
	if('-o' in sys.argv):
		output_dir = sys.argv[sys.argv.index('-o')]
		
	#select either global or local normalization

	input_dir = sys.argv[1]

	filenames = read_files_in_dir(input_dir)
	for i in filenames:
		print(i)
	t_s = time.time()

	uid_file = open("uid_file.csv", 'w')
	uid_file.write("uid, filename, label\n")
	for i in range(len(filenames)):
		label = filenames[i].split('/')[-2]
		uid_file.write(str(i)+','+filenames[i]+','+str(ITR_NAMES.index(label))+'\n')
	uid_file.close()

	for c3d_depth in range(5):
		new_dir = os.path.join(output_dir, str(c3d_depth))+'/'

		print("output placed in "+new_dir)
		if not os.path.exists(new_dir):
			os.makedirs(new_dir)
		
		#store thresholded info in temporary files in a a queue
		records = Queue()

		# get the maximum and mimimum activation values for each iad row
		file_reader = RosBagFileReader(filenames, batch_size =1)
		max_vals, min_vals = convert_videos_to_IAD(file_reader, c3d_depth, records)

		#need to reset graph after each map generation because graphs are read-only
		tf.reset_default_graph()

		# generate IADs using the earlier identified values
		global_norm(records, max_vals, min_vals, c3d_depth, new_dir)

		tf.reset_default_graph()

	print("completed_in: ", time.time()-t_s)

		