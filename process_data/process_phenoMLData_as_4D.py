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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-a', '--allp', action='store_true',
                       help='include all particle for all events into output file')
    group.add_argument('-j', '--jets-only', action='store_true',
                       help='include only jets from all events into output file')
    group.add_argument('-nj', '--not-jets', action='store_true',
                       help='nclude only non-jet particles from all events into output file')
    group.add_argument('-e', '--event-jets', action='store_true',
                       help='include only those events that contain only jets into output file')
    parser.add_argument('-w', '--wfile', nargs='?',
                         help='global path to processed file; must not include file type extension')
    return parser.parse_args()

# Function for formatting save file location
# Arguments:
#     input_path: global path to a file containing the data set
#     args: command line arguments, containing settings information
# Returns: string containing global path to output filename
def format_save_path(input_path, args):
    save_dir = os.path.dirname(input_path)
    input_filename, _ = os.path.splitext(os.path.basename(input_path))
    if (args.allp):
        return '{}/processed_4D_{}_all_events_all_particles'.format(save_dir, input_filename)
    elif (args.jets_only):
        return '{}/processed_4D_{}_all_events_but_only_jet_particles'.format(save_dir, input_filename)
    elif (args.not_jets):
        return '{}/processed_4D_{}_all_events_but_only_non_jet_particles'.format(save_dir, input_filename)
    elif (args.event_jets):
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

    return data

# Function to filter out events containing certain particles
# Arguments:
#     x: DataFrame containing events to be filtered
#     ignore: list containing particles events containing which should be ignored
# Returns: filtered events DataFrame
def filter_non_jet_events(x, ignore):
    ignore_list = []
    for i in range(len(x)):
        for j in x.loc[i].keys():
            if 'obj' in j:
                if x.loc[i][j] in ignore:
                    ignore_list.append(i)
                    break

    #print(ignore_list)

    return x.drop(ignore_list)

# Function to filter out non-jet particles from events
# Arguments:
#     x: DataFrame containing events to be filtered
# Returns: filtered events DataFrame
def filter_not_jets(x):
    lst = []
    for i in range(x.shape[0]):
        if  (x[i][0] == 'j') or (x[i][0] == 'b'):
            continue
        else:
            lst.append(i)
            #print(i, x[i][0])
    return np.delete(x, lst, 0)

# Function to filter out jet particles from events
# Arguments:
#     x: DataFrame containing events to be filtered
# Returns: filtered events DataFrame
def filter_jets(x):
    lst = []
    for i in range(x.shape[0]):
        if (x[i][0] == 'j') or (x[i][0] == 'b'):
            lst.append(i)
            #print(i, x[i][0])
    return np.delete(x, lst, 0)

def main():
    # Resolve command line arguments
    args = args_parser()
    input_path = args.rfile
    save_path = args.wfile if args.wfile else format_save_path(input_path, args)
    if (os.path.splitext(save_path)[1]):
        print('Invalid write file: write file must not include type extension')
        return

    # Get data from a file
    data = read_data(input_path, rlimit=200000)
    
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

    if args.event_jets:
        # Filter out events containing non-jet particles
        ignore_particles = ['e-', 'e+', 'm-', 'm+', 'g']
        x_df = filter_non_jet_events(x_df, ignore_particles)

    x = x_df.values.reshape([x_df.shape[0]*x_df.shape[1]//5,5])

    lst = []
    for i in range(x.shape[0]):
        if (x[i] == 0).all():
            lst.append(i)        
    x1 = np.delete(x, lst, 0)
    del x

    if args.jets_only or args.event_jets:
        # Filter out non jet particles
        data = filter_not_jets(x1)
    elif args.not_jets:
        # Filter out jet particles
        data = filter_jets(x1)
    else:
        data = x1

    print(len(data))

    col_names = ['obj', 'E', 'pt', 'eta', 'phi']

    data_df = pd.DataFrame(data, columns=col_names)
    data_df['obj'].to_pickle(save_path + '_meta_obj.pkl')

    data_df = data_df.drop(columns='obj')

    data_df = data_df.astype('float32')

    data_df.to_pickle(save_path + '_4D.pkl')
    
    return

if __name__=='__main__':
    main()
