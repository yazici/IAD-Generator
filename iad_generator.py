from file_io import read_file, generate_model_input
from iad_gen_global_norm import convert_videos_to_IAD, global_norm, read_files_in_dir

if __name__ == '__main__':

	assert len(sys.argv) >= 2, "Usage: python iad_generator.py <src_directory> -o <out_directory>"

	output_dir = sys.argv.index('-o')
	if(output_dir > 0):
		output_dir = sys.argv[output_dir]
	else:
		output_dir = "generated_iads/"
	#select either global or local normalization

	input_dir = sys.argv[1]

	filenames = read_files_in_dir(cur_dir)

	t_s = time.time()

	for c3d_depth in range(5):

		
		print("output placed in "+output_dir)
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		
		#store thresholded info in temporary files in a a queue
		records = Queue()

		# get the maximum and mimimum activation values for each iad row
		max_vals, min_vals = convert_videos_to_IAD(filenames, c3d_depth, records)

		#need to reset graph after each map generation because graphs are read-only
		tf.reset_default_graph()

		# generate IADs using the earlier identified values
		global_norm(records, max_vals, min_vals)

		tf.reset_default_graph()

	print("completed_in: ", time.time()-t_s)

		