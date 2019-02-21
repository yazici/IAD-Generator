import tensorflow as tf
import numpy as np

import os, random
from os.path import isfile, join, isdir

VERB_CLASSES = 125
NOUN_CLASSES = 331

time_window= 512


# I SHOULD LOOK AT THE generate_model_input FUNC for my earlier code

def generate_model_input(placeholders, data_source, sess):
	'''
	Reads a single entry from the specified data_source and stores the entry in
	the provided placeholder for use in sess
		-placeholders: a dictionary of placeholders
		-data_source: a tensor leading to a collection of TFRecords as would be 
			generated with the "input_pipeline" function
		-sess: the current session
	'''

	# read a single entry of the data_source into numpy arrays
	input_tensor = [sess.run(data_source)][0]

	np_values = {"img": input_tensor[0],
				"uid": input_tensor[1],
				"length": input_tensor[2][0],
				"verb": input_tensor[3],
				"noun": input_tensor[4]}
	
	# pad and reshape the img data for use with the C3D network
	if (np_values["length"] > time_window):
		return 0, 0

	data_ratio = float(np_values["length"])/time_window
	buffer_len = (time_window-np_values["length"])

	img_data = np_values["img"].reshape((np_values["length"], 112, 112,3))
	img_data = np.pad(img_data, 
									((0,buffer_len), (0,0), (0,0),(0,0)), 
									'constant', 
									constant_values=(0,0))
	img_data = np.expand_dims(img_data, axis=0)

	#place in dict

	placeholder_values = { placeholders: img_data}

	information_values = {
		"uid": np_values["uid"][0],
		"length": np_values["length"],
		"verb": np_values["verb"][0],
		"noun": np_values["noun"][0],
		"data_ratio": data_ratio}

	return placeholder_values, information_values

def parse_sequence_example(filename_queue):
	'''
	Reads a TFRecord and separate the output into its constituent parts
		-filename_queue: a filename queue as generated using string_input_producer
	'''
	reader = tf.TFRecordReader()
	_, example = reader.read(filename_queue)
	
	# Read value features (data, labels, and identifiers)
	context_features = {		
		"length": tf.FixedLenFeature([], dtype=tf.int64),
		"uid": tf.FixedLenFeature([], dtype=tf.int64),
		

		"verb": tf.FixedLenFeature([], dtype=tf.int64),
		"noun": tf.FixedLenFeature([], dtype=tf.int64)
	}

	# Read string features (input data)
	sequence_features = {
		"img_raw": tf.FixedLenSequenceFeature([], dtype=tf.string)
	}
	
	# Parse the example
	context_parsed, sequence_parsed = tf.parse_single_sequence_example(
		serialized=example,
		context_features=context_features,
		sequence_features=sequence_features
	)

	# Decode the byte data of the data sequences
	sequence_data = {
		"img_raw": tf.decode_raw(sequence_parsed["img_raw"], tf.uint8)# float values for IAD
	}
	
	return context_parsed, sequence_data

def input_pipeline(filenames, batch_size=1, randomize=False):
	'''
	Read TFRecords into usable arrays
		-filenames: an array of string filenames
		-randomize: a bool indicating whether the files should be read in a 
			randomized or sequential order
	'''

	# read the files one at a time
	filename_queue = tf.train.string_input_producer(
			filenames, shuffle=randomize)
	
	# --------------------
	# parse tfrecord files and define file as a single "data_tensor"

	context_parsed, sequence_parsed = parse_sequence_example(filename_queue)

	def extractFeature(input_tensor, cast=tf.int32):
		return tf.cast(tf.reshape(input_tensor, [-1]), cast)
	
	# extract data sequences from tfrecord tensors
	data_tensor = [
		extractFeature(sequence_parsed["img_raw"], cast=tf.uint8),
		context_parsed["uid"],
		context_parsed["length"],
		context_parsed["verb"],
		context_parsed["noun"]
	]

	# --------------------
	# store the data_tensors in a queue that can be read from

	dtypes = list(map(lambda x: x.dtype, data_tensor))
	shapes = list(map(lambda x: x.get_shape(), data_tensor))

	min_after_dequeue = 7 
	capacity = min_after_dequeue + 3 

	queue=None
	if(randomize):
		queue = tf.RandomShuffleQueue(capacity, min_after_dequeue, dtypes)
	else:
		queue = tf.FIFOQueue(capacity, dtypes)

	num_threads = 1
	enqueue_op = queue.enqueue(data_tensor)
	qr = tf.train.QueueRunner(queue, [enqueue_op] * num_threads)


	QUEUE_RUNNERS = 1
	tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
	outputs = queue.dequeue()

	for tensor, shape in zip(outputs, shapes):
		tensor.set_shape(shape)
	
	# --------------------
	# define how many files should be read from each batch

	batch_size = 1
	batched_output = tf.train.batch(outputs, 
			batch_size, 
			capacity=capacity,
			dynamic_pad=True
			)
									
	return batched_output



def read_files_in_dir(directory, randomize=False, limit_dataset=0, recursive=True):
	'''
	A helper function for locating all of the files in a directory. Returns the files in
	as a TF list. also returns the size of the dataset.
		-directory: the directory containing the TFRecords
		-randomize: a bool indicating whether the data should be randomized or not
		-limit_dataset: an integer to limit the size of teh dataset
		-recursive: a bool indicating whether the directory contains other directories
	'''

	# identify which folders contain files
	folders = []
	if(recursive):
		folders = [join(directory, f) for f in os.listdir(directory) if isdir(join(directory, f))]	
		folders.sort()
		if(limit_dataset > 0):
			folders = folders[:limit_dataset]
	else:
		folders = [directory]

	# identify all of the files in all of the folders
	all_files = []
	for folder in folders:
		all_files += [join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f)) ]

	# randomize the order in which the files are listed
	if(randomize):
		random.shuffle(all_files)

	return input_pipeline(all_files, randomize=randomize), len(all_files)

def read_files_in_dir_group(directory, randomize=False, limit_dataset=0, selection=[1]):
	'''
	A helper function for locating all of the files in a directory. Returns the files in
	as a TF list. also returns the size of the dataset.
		-directory: the directory containing the TFRecords
		-randomize: a bool indicating whether the data should be randomized or not
		-limit_dataset: an integer to limit the size of teh dataset
		-recursive: a bool indicating whether the directory contains other directories
	'''

	# identify which folders contain files
	folders = []

	'''
	if(recursive):
		folders = [join(directory, f) for f in os.listdir(directory) if isdir(join(directory, f))]	
		folders.sort()
		if(limit_dataset > 0):
			folders = folders[:limit_dataset]
	else:
		folders = [directory]
	'''
	folders = []
	for i in range(len(selection)):
		if (selection[i]):
			folders.append(directory+str(i))

	# identify all of the files in all of the folders
	all_files = []
	for folder in folders:
		all_files += [join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f)) ]

	# randomize the order in which the files are listed
	if(randomize):
		random.shuffle(all_files)

	return input_pipeline(all_files, randomize=randomize), len(all_files)