import os
import click
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

@click.command()
@click.option('--input_path',
              type=click.STRING,
              default='/afs/cern.ch/work/h/hgupta/public/phenoML/datasets/ttbar_10fb.csv',
              help='The path to the csv file.')

@click.option('--save_path',
              type=click.STRING,
              default='/afs/cern.ch/work/h/hgupta/public/phenoML/datasets/processed_4D_ttbar_10fb_all_events_but_only_jet_particles',
              help='The name to the pkl file.')

def csv_to_df(input_path, save_path):
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
    csv_to_df()
