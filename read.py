# https://cloud.google.com/storage/docs/reference/libraries#setting_up_authentication

import sys, csv, json, os, time

import s3fs
import pandas as pd
from google.cloud import storage
import warnings
warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

os.environ['GOOGLE_APPLICAITON_CREDENTIALS'] = "credentials.json"

project_name = 'algo-trader-staging'

def read():
    print('reading...')

    try:
        storage_client = storage.Client()
        df = pd.read_csv('gs://algo-trader-staging/datasets/XRP_USDT/5m/50_candles/dataset_XRP_USDT_5m_50candles_04-17-2022_03-04.csv')
        
        print('df', df)
        # df = pd.DataFrame(data=[{1,2,3},{4,5,6}],columns=['a','b','c'])
        # bucket = storage_client.get_bucket('algo-trader-staging')
        # bucket.blob('upload_test/test.csv').upload_from_string(df.to_csv(), 'text/csv')

        return True
    except Exception as e:
        print(e)
        return False

def write():
    print('writing...')

    try:
        storage_client = storage.Client()
        
        df = pd.DataFrame(data=[{1,2,3},{4,5,6}],columns=['a','b','c'])
        bucket = storage_client.get_bucket('algo-trader-staging')
        bucket.blob('upload_test/test.csv').upload_from_string(df.to_csv(), 'text/csv')

        return True
    except Exception as e:
        print(e)
        return False

if __name__ == "__main__":
    read()