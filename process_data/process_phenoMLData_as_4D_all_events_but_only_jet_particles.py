import os
import argparse
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Function for parsing command line arguments
# Refturns: composite Object containing command line arguments
def args_parser():
    parser = argparse.ArgumentParser(description='Pre-process csv data: convert to pickle format and choose appropriate particles')
    # User must provide arguments for training and testing datasets
    required = parser.add_argument_group('required named arguments')
    # If only flag is given, const value of each argument will be used
    required.add_argument('-r', '--rfile', nargs='?', required=True,
                          const='/nfs/atlas/mvaskev/sm/z_jets_10fb.csv',
                          help='global path to dataset file; must be in text (.csv) format')
    required.add_argument('-w', '--wfile', nargs='?', required=True,
                          const='/nfs/atlas/mvaskev/sm/processed_4D_z_jets_10fb_all_events_but_only_jet_particles',
                          help='global path to processed file; must not include file type extension')
    return parser.parse_args()

def main():
    # Resolve command line arguments
    args = args_parser()
    input_path, save_path = args.rfile, args.wfile

    data = []    
    print('Reading data at ', input_path)
    with open(input_path, 'r') as file:
        for line in file.readlines():
            line = line.replace(';', ',')
            line = line.rstrip(',\n')
            line = line.split(',')
            data.append(line)

    data = data[:20000]

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

    x_train_df = pd.DataFrame(df.values, columns=col_names)
    x_train_df.fillna(value=0, inplace=True)

    meta_train_df = x_train_df[meta_cols]
    meta_train_df.to_pickle(save_path + '_metaData.pkl')

    x_train_df = x_train_df.drop(columns=meta_cols)

    x = x_train_df_1.values.reshape([x_train_df_1.shape[0]*x_train_df_1.shape[1]//5,5])

    lst = []
    for i in range(x.shape[0]):
        if (x[i] == 0).all():
            lst.append(i)        
    x1 = np.delete(x, lst, 0)
    del x

    lst = []
    for i in range(x1.shape[0]):   
        if  (x1[i][0] == 'j') or (x1[i][0] == 'b'):
            continue
        else:
            lst.append(i)
            print(i, x1[i][0])
    data_train = np.delete(x1, lst, 0)
    print(len(data_train))

    col_names = ['obj', 'E', 'pt', 'eta', 'phi']

    # data_train_df = pd.DataFrame(data_train, columns=col_names)
    # data_train_df['obj'].to_pickle(save_path + '_meta_obj_train.pkl')

    # data_test_df = pd.DataFrame(data_test, columns=col_names)
    # data_test_df['obj'].to_pickle(save_path + '_meta_obj_test.pkl')

    data_train_df = pd.DataFrame(data_train, columns=col_names)
    data_train_df['obj'].to_pickle(save_path + '_meta_obj.pkl')

    data_train_df = data_train_df.drop(columns='obj')
    # data_test_df = data_test_df.drop(columns='obj')

    data_train_df = data_train_df.astype('float32')
    # data_test_df = data_test_df.astype('float32')

    # data_train_df.to_pickle(save_path + '_4D_train.pkl')
    # data_test_df.to_pickle(save_path + '_4D_test.pkl')

    data_train_df.to_pickle(save_path + '_4D.pkl')
    
    return

if __name__=='__main__':
    main()
