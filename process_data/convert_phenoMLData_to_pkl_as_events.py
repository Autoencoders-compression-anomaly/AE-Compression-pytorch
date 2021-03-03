import os
import argparse
import random
import pandas as pd
import numpy as np
import pickle as pkl

# Function for parsing command line arguments
# Returns: composite Object containing command line arguments
def args_parser():
    parser = argparse.ArgumentParser(description='Pre-process csv data: convert to pickle format and choose appropriate particles')
    # User must provide global path to data file
    parser.add_argument('rfile', nargs=1,
                        help='global path to dataset file; must be in csv format')
    return parser.parse_args()

# Helper function to create directory if such does not exist
# Arguments:
#     directory: string containing path to directory to be ceated
def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function for formatting save file location
# Arguments:
#     input_path: global path to a file containing the data set
# Returns: string containing global path to output filename
def format_save_path(input_path):
    save_dir = os.path.dirname(input_path)
    input_filename, _ = os.path.splitext(os.path.basename(input_path))
    gname = 'processed_events_{}'.format(input_filename)
    sdir = '{}/{}/'.format(save_dir, gname)
    make_directory(sdir)
    return '{}{}.pkl'.format(sdir, gname)

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
    print('Reading data at {}'.format(input_path))
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
    return data[:plimit], not lbreak

def main():
    args = args_parser()
    input_path = args.rfile[0]
    save_path = format_save_path(input_path)
    print(save_path)

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

        df = df.fillna(0)
    
        variables = [entry for entry in df.columns if entry[0] == 'E'] + [entry for entry in df.columns if entry[0:2] == 'pt'] + [entry for entry in df.columns if entry[0:2] == 'et'] + [entry for entry in df.columns if entry[0:2] == 'ph']

        df = df[['process_ID']+variables]
        one_hot = pd.get_dummies(df['process_ID'])
        processes = one_hot.columns
        df.drop('process_ID', axis = 'columns', inplace = True)
        df = pd.concat([df, one_hot], sort = False, axis = 1)

        print('Creating data files at {}'.format(save_path))

        #Pickle the dataframe to keep it fresh
        if save_path:
            with open(save_path, 'ab') as f:
                pkl.dump(df, f)

        print('Create data files done')

        # Forward the reading and writing variables
        fnumber = fnumber + 1
        rbegin = rbegin + rstep + 1
        rlimit = rlimit + rstep
 
        # Reset variables
        del df
    
    return

if __name__=='__main__':
    main()
