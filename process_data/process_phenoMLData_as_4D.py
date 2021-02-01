import os
import argparse
import random
import h5py
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split

# Function for parsing command line arguments
# Returns: composite Object containing command line arguments
def args_parser():
    parser = argparse.ArgumentParser(description='Pre-process csv data: convert to pickle format and choose appropriate particles')
    # User must provide global path to data file
    parser.add_argument('rfile', nargs=1,
                        help='global path to dataset file; must be in csv format')
    parser.add_argument('-f', '--file-type', nargs='?', default='pkl', const='pkl',
                        choices=['pkl', 'h5'],
                        help='choose a file type for a processed data file (default: %(default)s)')
    parser.add_argument('-s', '--setting', nargs='?', default='all', const='all',
                        choices=['all', 'jets', 'non_jets', 'light_jets', 'b_jets', 'jet_events'],
                        help='choose a setting upon which to filter particles (default: %(default)s)')
    return parser.parse_args()

# Helper function to create directory if such does not exist
# Arguments:
#     directory: string containing path to directory to be created
def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function for formatting save file location
# Arguments:
#     input_path: global path to a file containing the data set
#     setting: command line argument choice, containing settings information
#     ext: command line argument choice for file type, in string format
# Returns: string containing global path to output filename
def format_save_path(input_path, setting, ext):
    save_dir = os.path.dirname(input_path)
    input_filename, _ = os.path.splitext(os.path.basename(input_path))
    if setting in ['all']:
        gname = 'processed_4D_{}_all_events_{}'.format(input_filename, setting)
    elif setting in ['jets', 'non_jets', 'light_jets', 'b_jets']:
        gname = 'processed_4D_{}_all_events_only_{}'.format(input_filename, setting)
    elif setting in ['jet_events']:
        gname = 'processed_4D_{}_{}_all'.format(input_filename, setting)
    sdir = '{}/{}/'.format(save_dir, gname)
    make_directory(sdir)
    if ext == 'pkl':
        return '{}{}_meta_data.{}'.format(sdir, gname, ext), \
               '{}{}_meta_obj.{}'.format(sdir, gname, ext), \
               '{}{}_4D.{}'.format(sdir, gname, ext)
    elif ext == 'h5':
        return ['{}{}.{}'.format(sdir, gname, ext)]

# Function for reading data input
# Arguments:
#     input_path: string containing global path to input file
#     rlimit: integer maximum number of lines to read from input file
#     rbegin: integer value line at which to start reading file
#     plimit: integer maximum number of particles to return
# Returns: list object with input data
def read_data(input_path, rlimit=None, rbegin=0, plimit=None):
    data = []
    lbreak = False
    print('Reading data at ', input_path)
    with open(input_path, 'r') as f:
        for cnt, line in enumerate(f):
            if cnt < rbegin:
                continue
            line = line.replace(';', ',')
            line = line.rstrip(',\n')
            line = line.split(',')
            data.append(line)
            if rlimit and cnt == rlimit:
                lbreak = True
                break
    return data[:plimit], lbreak

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
    input_path = args.rfile[0]
    save_paths = format_save_path(input_path, args.setting, args.file_type)

    rcontinue = True
    fnumber = 0
    rstep = 1000000
    rlimit = rstep
    rbegin = 0

    while rcontinue:
        print('Begin batch {}'.format(fnumber))

        # Get data from a file
        data, rcontinue = read_data(input_path, rlimit=rlimit, rbegin=rbegin)
        print('Read data done')
    
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
        print('Processing data')
        df = pd.DataFrame(data, columns=col_names)
        df.fillna(value=np.nan, inplace=True)

        x_df = pd.DataFrame(df.values, columns=col_names)
        x_df.fillna(value=0, inplace=True)

        meta_df = x_df[meta_cols]
        if args.file_type == 'pkl':
            with open(save_paths[0], 'ab') as f:
                pkl.dump(meta_df, f)

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

        print('Process data done')
        print('Number of particles: {}'.format(len(data)))

        col_names = ['obj', 'E', 'pt', 'eta', 'phi']

        print('Creating data files')

        data_df = pd.DataFrame(data, columns=col_names)
        if args.file_type == 'pkl':
            with open(save_paths[1], 'ab') as f:
                pkl.dump(data_df['obj'], f)

        data_df = data_df.drop(columns='obj')

        data_df = data_df.astype('float32')

        if args.file_type == 'pkl':
            with open(save_paths[2], 'ab') as f:
                pkl.dump(data_df, f)
        elif args.file_type == 'h5':
            with h5py.File(save_paths[0], 'a') as f:
                if fnumber == 0:
                    data = np.delete(data, 0, 1)
                    dset = f.create_dataset('data', data.shape, maxshape=(None, data.shape[1]), chunks=True)
                    dset[:] = data.astype('float32')
                else:
                    dset = f['data']
                    dset.resize(dset.shape[0]+data.shape[0], axis=0)
                    data = np.delete(data, 0, 1)
                    dset[-data.shape[0]:] = data.astype('float32')

        print('Create data files done')

        # Forward the reading and writing variables
        fnumber = fnumber + 1
        rbegin = rbegin + rstep + 1
        rlimit = rlimit + rstep

        # Reset variables
        del data
        del data_df
    
    return

if __name__=='__main__':
    main()
