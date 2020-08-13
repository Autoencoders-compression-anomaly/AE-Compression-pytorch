import os
import click
import random
import pandas as pd
import numpy as np

@click.command()
@click.option('--input_path',
              type=click.STRING,
              default='/home/honey/cern/July/datasets/njets_10fb.csv',
              help='The path to the csv file.')

@click.option('--save_path',
              type=click.STRING,
              default='/home/honey/cern/July/datasets/processed_tiny_njets_10fb.pkl',
              help='The path to the pkl file.')

def csv_to_df(input_path, save_path):
    data = []    
    print('Reading data at ', input_path)
    with open(input_path, 'r') as file:
        for line in file.readlines():
            line = line.replace(';', ',')
            line = line.rstrip(',\n')
            line = line.split(',')
            data.append(line)
    
    # Shorten data - for debug purposes
    # data_ = []
    # idxs = random.sample(range(0, len(data)), int(0.1 * len(data)))
    # for idx in idxs:
        # data_.append(data[idx])
    # data = data_

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

    #Pickle the dataframe to keep it fresh
    if save_path:
        print('Saving at ', save_path)
        df.to_pickle(save_path)
    
    return df

if __name__=='__main__':
    csv_to_df()