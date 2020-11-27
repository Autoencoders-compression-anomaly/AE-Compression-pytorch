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
    # User must provide global path to data file
    parser.add_argument('rfile', nargs=1,
                        help='global path to dataset file; must be in csv format')
    parser.add_argument('-s', '--setting', nargs='?', default='all', const='all',
                        choices=['all', 'jets', 'non_jets', 'light_jets', 'b_jets', 'jet_events'],
                        help='choose a setting upon which to filter particles (default: %(default)s)')
    parser.add_argument('-w', '--wfile', nargs='?',
                         help='global path to processed file; must not include file type extension')
    return parser.parse_args()

# Function for formatting save file location
# Arguments:
#     input_path: global path to a file containing the data set
#     setting: command line argument choice, containing settings information
# Returns: string containing global path to output filename
def format_save_path(input_path, setting):
    save_dir = os.path.dirname(input_path)
    input_filename, _ = os.path.splitext(os.path.basename(input_path))
    if setting == 'all':
        return '{}/processed_4D_{}_all_events_all_particles'.format(save_dir, input_filename)
    elif setting == 'jets':
        return '{}/processed_4D_{}_all_events_but_only_jet_particles'.format(save_dir, input_filename)
    elif setting == 'non_jets':
        return '{}/processed_4D_{}_all_events_but_only_non_jet_particles'.format(save_dir, input_filename)
    elif setting == 'light_jets':
        return '{}/processed_4D_{}_all_events_but_only_light_jet_particles'.format(save_dir, input_filename)
    elif setting == 'b_jets':
        return '{}/processed_4D_{}_all_events_but_only_b_jet_particles'.format(save_dir, input_filename)
    elif setting == 'jet_events':
        return '{}/processed_4D_{}_events_with_only_jet_particles'.format(save_dir, input_filename)

# Function for reading data input
# Arguments:
#     input_path: string containing global path to input file
#     rlimit: integer maximum number of lines to read from input file
#     plimit: integer maximum number of particles to return
# Returns: list object with input data
def read_data(input_path, rlimit=None, plimit=None):
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

# Function to filter out non-b-jet particles from events
# Arguments:
#     x: DataFrame containing events to be filtered
# Returns: filtered events DataFrame
def filter_not_b_jets(x):
    lst = []
    for i in range(x.shape[0]):
        if (x[i][0] == 'b'):
            continue
        else:
            lst.append(i)
    return np.delete(x, lst, 0)

# Function to filter out non-light-jet particles from events
# Arguments:
#     x: DataFrame containing events to be filtered
# Returns: filtered events DataFrame
def filter_not_light_jets(x):
    lst = []
    for i in range(x.shape[0]):
        if (x[i][0] == 'j'):
            continue
        else:
            lst.append(i)
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
    save_path = args.wfile if args.wfile else format_save_path(input_path, args.setting)
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

    if args.setting == 'jet_events':
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

    if args.setting == 'jets' or args.setting == 'jet_events':
        # Filter out non jet particles
        data = filter_not_jets(x1)
    elif args.setting == 'non_jets':
        # Filter out jet particles
        data = filter_jets(x1)
    elif args.setting == 'light_jets':
        # Filter out all particles but light jet ones
        data = filter_not_light_jets(x1)
    elif args.setting == 'b_jets':
        # Filter out all particles but b-jet ones
        data = filter_not_b_jets(x1)
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
