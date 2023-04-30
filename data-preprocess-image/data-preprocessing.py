#!/usr/bin/env python3
import platform; print(platform.platform())
import sys; print("Python", sys.version)

import numpy as np # linear algebra
import pandas as pd; print("pandas version:",pd.__version__) # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from google.cloud import storage

import warnings
warnings.filterwarnings('ignore')

import os
import argparse
from pathlib import Path
import json

#### -------------------- data pre-processing -----------------------------
def load_data_from_gcs(input_data_dir,bucket_name,relative_path,save_file_name): 
    '''download data from google cloud storage and save to local path'''
    print('='*40)
    print('data loading started for {fname} from gcs'.format(fname=save_file_name))
    storage_client = storage.Client()
    public_bucket = storage_client.bucket(bucket_name)
    blob = public_bucket.blob(relative_path)
    blob.download_to_filename('{input_dir}/{save_file}'.format(input_dir = input_data_dir, save_file=save_file_name))
    print('data loading completed for {fname} from gcs'.format(fname=save_file_name))
    print('='*40)

def load_data_to_df(input_data_dir,file_name): 
    '''load csv files in local machine to pandas data frames'''
    print('data loading started from csv to pandas' )
    df = pd.read_csv('{input_dir}/{fname}'.format(input_dir = input_data_dir,fname=file_name),header=None, delimiter=r"\s+") 
    
    
    print('data loading completed from csv to pandas' )
    print('='*40)
    
    return df

def load_json_data(input_data_dir,file_name): 
    '''load json files in local machine to dictionaries'''
    print('config loading started')
    config_dir_final = os.path.join(input_data_dir, file_name)
    f = open(config_dir_final)
    config = json.load(f)
    f.close()
    
    print('config loading completed' )
    print('='*40)
    
    return config


def data_pre_processing(house_price_df,col_names,select_cols,main_data_output):
    '''perform basic pre-processing steps'''
    house_price_df.columns = col_names
    house_price_df.head()
    house_price_df.shape
    house_price_df.isna().sum()


    data_final = house_price_df[select_cols]

    main_output_final = os.path.join(main_data_output, "main_data_preprocessed.csv") 
    data_final.to_csv(main_output_final)



if __name__ == '__main__':    
    # Defining and parsing the command-line arguments : to be used in component
    parser = argparse.ArgumentParser(description='Boston Hoise Price Model data pre processing')
# Paths must be passed in, not hardcoded
    
    parser.add_argument('--input1-path', type=str,
    help='Cloud bucket name containing raw data')
    
    parser.add_argument('--input2-path', type=str,
      help='Relative path containing house_price.csv')
    parser.add_argument('--input3-path', type=str,
      help='Relative path containing config.json')

    parser.add_argument('--input4-path', type=str,
      help='Save file name for house_price.csv')
    parser.add_argument('--input5-path', type=str,
      help='Save file name for config.json')
    

    parser.add_argument('--output1-path', type=str,
      help='Path of the local folder where the main_data_preprocessed.csv should be written.')
    
    args = parser.parse_args()
    
    # Creating the directory where the output file is created (the directory
    # may or may not exist).
    Path(args.output1_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.output1_path):
        os.makedirs(args.output1_path)
    
    ####--------------------relative paths ----------------------
    DATA_DIR= "data"
    CONFIG_DIR = "config"
    
    input_data_dir = "{}/input".format(DATA_DIR)
    train_test_dir="{}/train_test".format(DATA_DIR)
    
    
    bucket_name = args.input1_path
    
    main_data_rel = args.input2_path
    config_data_rel = args.input3_path

    house_price_file_name = args.input4_path
    config_file_name = args.input5_path

    main_data_output = args.output1_path
    

    ####-----------------------------------------------------------------------
    print('='*40)
    print('data-preprocessing.py script started')

    ##load house price data
    load_data_from_gcs(input_data_dir,bucket_name,main_data_rel,house_price_file_name)

    ##load config data
    load_data_from_gcs(CONFIG_DIR,bucket_name,config_data_rel,config_file_name)


    house_price_df = load_data_to_df(input_data_dir,house_price_file_name)
    config = load_json_data(CONFIG_DIR,config_file_name)

    col_names = config['raw_df_cols']
    select_cols = config['select_cols'] + config['response_var']
    data_pre_processing(house_price_df,col_names,select_cols,main_data_output)

    print('data-preprocessing.py script completed')
    print('='*40)
    
    
  