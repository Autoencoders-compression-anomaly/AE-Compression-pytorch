import os
import argparse
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Function for parsing command line arguments
# Returns: composite Object containing command line arguments
def args_parser():
    parser = argparse.ArgumentParser(description='Pre-process csv data: convert to pickle format and choose appropriate particles')
    # User must provide arguments for training and testing datasets
    required = parser.add_argument_group('required named arguments')
    # If only flag is given, const value of each argument will be used
    required.add_argument('-r', '--rfile', nargs='?', required=True,
                          const='/nfs/atlas/mvaskev/sm/z_jets_10fb.csv',
                          help='global path to dataset file; must be in csv format')
    parser.add_argument('-w', '--wfile', nargs='?',
                         help='global path to processed file; must not include file type extension')
    return parser.parse_args()

# Function for formatting save file location
# Arguments:
#     input_path: global path to a file containing the data set
def format_save_path(input_path):
    save_dir = os.path.dirname(input_path)
    input_filename, _ = os.path.splitext(os.path.basename(input_path))
    return '{}/processed_4D_{}_events_with_only_jet_particles'.format(save_dir, input_filename)

# Function for reading data input
# Arguments:
#     input_path: string containing global path to input file
#     rlimit: integer maximum number of lines to read from input file
#     plimit: integer maximum number of particles to return
# Returns: list object with input data
def read_data(input_path, rlimit=None, plimit=20000):
    data = []
    print('Reading data at ', input_path)
    with open(input_path, 'r') as f:
        for cnt, line in enumerate(f):
            line = line.replace(';', ',')
            line = line.rstrip(',\n')
            line = line.split(',')
            data.append(line)
            if rlimit and cnt == rlimit:
                break

    return data[:plimit]

# Function to filter out non-jet particles from events
# Arguments:
#     x1: DataFrame containing events to be filtered
def filter_not_jets(x1):
    lst = []
    for i in range(x1.shape[0]):
        if  (x1[i][0] == 'j') or (x1[i][0] == 'b'):
            continue
        else:
            lst.append(i)
            print(i, x1[i][0])
    return np.delete(x1, lst, 0)

def main():
    # Resolve command line arguments
    args = args_parser()
    input_path = args.rfile
    save_path = args.wfile if args.wfile else format_save_path(input_path)
    if (os.path.splitext(save_path)[1]):
        print('Invalid write file: write file must not include type extension')
        return

    data = read_data(input_path, rlimit=3)
    
    #Find the longest line in the data 
    longest_line = max(data, key = len)

    #Set the maximum number of columns
    max_col_num = len(longest_line)

    #Set the columns names
    col_names = ['event_ID', 'process_ID', 'event_weight', 'MET', 'MET_Phi']
    meta_cols = col_names.copy()

    for i in range(1, (int((max_col_num-5)/5))+1):
        col_names.append('obj'+str(i))
        col_names.append('E'+str(i))
        col_names.append('pt'+str(i))
        col_names.append('eta'+str(i))
        col_names.append('phi'+str(i))

    #Create a dataframe from the list, using the column names from before

    print('Processing the data..')
    df = pd.DataFrame(data, columns=col_names)
    df.fillna(value=np.nan, inplace=True)

    x_df = pd.DataFrame(df.values, columns=col_names)
    x_df.fillna(value=0, inplace=True)

    meta_df = x_df[meta_cols]
    meta_df.to_pickle(save_path + '_metaData.pkl')

    x_df = x_df.drop(columns=meta_cols)

    ignore_particles = ['e-', 'e+', 'm-', 'm+', 'g']
    ignore_list = []
    for i in range(len(x_df)):
        for j in x_df.loc[i].keys():
            if 'obj' in j:
                if x_df.loc[i][j] in ignore_particles:
                    ignore_list.append(i)
                    break

    print(ignore_list)

    x_df = x_df.drop(ignore_list)

    x = x_df.values.reshape([x_df.shape[0]*x_df.shape[1]//5,5])

    lst = []
    for i in range(x.shape[0]):
        if (x[i] == 0).all():
            lst.append(i)        
    x1 = np.delete(x, lst, 0)
    del x

    data_train = filter_not_jets(x1)
    print(len(data_train))

    col_names = ['obj', 'E', 'pt', 'eta', 'phi']

    # data_df = pd.DataFrame(data_train, columns=col_names)
    # data_df['obj'].to_pickle(save_path + '_meta_obj_train.pkl')

    # data_test_df = pd.DataFrame(data_test, columns=col_names)
    # data_test_df['obj'].to_pickle(save_path + '_meta_obj_test.pkl')

    data_df = pd.DataFrame(data_train, columns=col_names)
    data_df['obj'].to_pickle(save_path + '_meta_obj.pkl')

    data_df = data_df.drop(columns='obj')
    # data_test_df = data_test_df.drop(columns='obj')

    data_df = data_df.astype('float32')
    # data_test_df = data_test_df.astype('float32')

    # data_df.to_pickle(save_path + '_4D_train.pkl')
    # data_test_df.to_pickle(save_path + '_4D_test.pkl')

    data_df.to_pickle(save_path + '_4D.pkl')
    
    return

if __name__=='__main__':
    main()
