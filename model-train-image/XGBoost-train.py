#!/usr/bin/env python3
import platform; print(platform.platform())
import sys; print("Python", sys.version)

import numpy as np # linear algebra
import pandas as pd; print("pandas version:",pd.__version__) # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import pickle

from google.cloud import storage

import warnings
warnings.filterwarnings('ignore')

import os
import argparse
from pathlib import Path
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb ; print("XGBoost Version", xgb.__version__)

from sklearn.metrics import mean_squared_error
from math import sqrt


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
    df = pd.read_csv('{input_dir}/{fname}'.format(input_dir = input_data_dir,fname=file_name)) 
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


def model_training(data,config):
    '''train XGBoost model'''
    select_cols = config['select_cols']
    response_var = config['response_var'][0]
    X = data[select_cols]
    y = data[response_var]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, shuffle=False)

    params=config['xgb_params']
    num_boost_round = config['num_boost_round']

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")],
        early_stopping_rounds=50
        #verbose_eval = False #### reduce prints in output
    )

    MAE = mean_absolute_error(model.predict(dtest), y_test)
    RMSE = sqrt(mean_squared_error(model.predict(dtest), y_test))

    return MAE,RMSE,model


def upload_data_to_gcs(bucket_name,relative_path,gcp_file_name,local_path): 
    '''upload files GCS from local machine'''
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    #blob = bucket.blob('{fname}'.format(fname=gcp_file_name))
    print('{path}/{fname}'.format(path=relative_path,fname=local_path))
    blob = bucket.blob('{path}/{fname}'.format(path=relative_path,fname=local_path))
    print(local_path)
    blob.upload_from_filename(local_path)
    print('-----Files uploaded to GCS successfully-----')
    print('='*40)





if __name__ == '__main__':    
    # Defining and parsing the command-line arguments : to be used in component
    parser = argparse.ArgumentParser(description='Boston House Price Model training')
# Paths must be passed in, not hardcoded
    
    parser.add_argument('--input1-path', type=str,
    help='Cloud bucket name containing raw data')
    
    parser.add_argument('--input2-path', type=str,
      help='Relative path main_data_preprocessed.csv')
    parser.add_argument('--input3-path', type=str,
      help='Relative path containing config.json')

    parser.add_argument('--input4-path', type=str,
      help='Save file name for config.json')

    parser.add_argument('--output1-path', type=str,
      help='Path of the local folder where model artifacts should be written.')

    parser.add_argument('--param1', type=str, 
      help='Pre-processed main file name')

    parser.add_argument('--param2', type=str, 
      help='Relative path of the GCS location to upload model artifacts')
    
    args = parser.parse_args()
    
    # Creating the directory where the output file is created (the directory
    # may or may not exist).

    if not os.path.exists(args.output1_path):
        os.makedirs(args.output1_path)
    
    ####--------------------relative paths ----------------------
    DATA_DIR= "data"
    CONFIG_DIR = "config"
    
    input_data_dir = "{}/input".format(DATA_DIR)
    
    
    bucket_name = args.input1_path
    
    main_data_rel = args.input2_path
    config_data_rel = args.input3_path

    config_file_name = args.input4_path

    main_pre_file_name = args.param1

    relative_path = args.param2

    model_sav_output = args.output1_path

    ####-----------------------------------------------------------------------
    print('='*40)
    print('XGBoost-train.py script started')


    ##load config data
    load_data_from_gcs(CONFIG_DIR,bucket_name,config_data_rel,config_file_name)


    main_data = load_data_to_df(main_data_rel,main_pre_file_name)
    config = load_json_data(CONFIG_DIR,config_file_name)

    MAE,RMSE,final_model= model_training(main_data,config)

    print("MAE for test set :",MAE)
    print("RMSE for test set :",RMSE)

    pick_file_name = "boston_house_price_model_"+datetime.today().strftime('%Y_%m_%d')+".sav"
    filename = '{path}/{fname}'.format(path = model_sav_output,fname=pick_file_name)
    pickle.dump(final_model, open(filename, 'wb'))
    print('='*40)
    print('House price model artifacts saved successfully')

    upload_data_to_gcs(bucket_name,relative_path,pick_file_name,filename)

    print('XGBoost-train.py script completed')
    print('='*40)
    
    
  