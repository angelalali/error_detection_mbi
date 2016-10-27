from os.path import join

# specify data folder
data_folder = '/Users/yisili/Documents/IBM/projects/kraft heinz project/data 2/'

# specify preprocessing files


# specify input files
# header_file = join(data_folder, 'data_header.csv')
column_description_file = join(data_folder, 'Fields Required for MDM Algorithms.xlsx')
# data_file = '/Users/yisili/Documents/IBM/projects/kraft heinz project/data 2/IBM_Materials_8318_8366_8330_8329.xlsx'
### partial data about 5k
# data_file = '/Users/yisili/Documents/IBM/projects/kraft heinz project/data 2/10-24 data partial.xlsx'
### all the data 75k data
data_file = '/Users/yisili/Documents/IBM/projects/kraft heinz project/data 2/10-20 15k.xlsx'

# specify output files
output_folder = '/Users/yisili/Documents/IBM/projects/kraft heinz project/output'
global_histogram_errors_file = join(output_folder, 'global_histogram_errors.csv')
decision_tree_errors_file = join(output_folder, 'decision_tree_errors.csv')
all_errors_file = join(output_folder, 'James_out.csv')
