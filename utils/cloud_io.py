import csv, json, os, time
import pandas as pd

from utils.helpers import BinanceHelper
from google.cloud import storage

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "google-credentials.json"
bucket_name = os.environ["GCP_CLOUD_STORAGE_BUCKET"]

def fetch_model():
    try: 
        filename_gcp = f"gs://{bucket_name}/datasets/{pair}/{interval}/{setup_features_and_labels}_candles/{filename}"
        return pd.read_csv(filename_gcp)
    except Exception as e:
        print(e)
        return False


def save_model(pair, filename_model, model, cloudStorage):
    if cloudStorage:
        path = f"gs://{bucket_name}/models/{pair}"
        model.save(f"{path}/{filename_model}")
    else:
        path = f"./models/{pair}"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        model.save(f"./models/{pair}/{filename_model}")

def fetch_dataset(pair, interval, candle_lookback_length):
    helper = BinanceHelper(
        pairs_of_interest=[pair],
        intervals_of_interest=[interval],
        candle_lookback_length=candle_lookback_length
    )

    [_, dataset] = helper.generate_datasets()

    return dataset

def fetch_latest_candles(pair, interval, candle_lookback_length):
    helper = BinanceHelper(
        pairs_of_interest=[pair],
        intervals_of_interest=[interval],
        candle_lookback_length=candle_lookback_length
    )
 
    klines = helper.generate_klines(
        pair_str=pair.replace("/",""),
        interval=interval,
        pair_as_key=pair.replace("/","_")
    )

    return helper.clean_data(klines)


def write_dataset_csvs(datasets, interfaceHelper, cloudStorage):
    try:
        storage_client = storage.Client()
        bucket_name = os.environ["GCP_CLOUD_STORAGE_BUCKET"]
        bucket = storage_client.get_bucket(bucket_name)

        for pair in datasets:
            dataset_obj = datasets[pair]
            
            for curr_set in dataset_obj["sets"]:
                dataset = curr_set["dataset"]
                interval = curr_set["interval"]
                path = f"./datasets/{pair}/{interval}/{interfaceHelper.candle_lookback_length}_candles"

                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)

                month = dataset.iloc[0]["month"]
                month = month if month > 9 else f"0{month}"
                day = dataset.iloc[0]["day"]
                day = day if day > 9 else f"0{day}"
                year = dataset.iloc[0]["year"]
                hour = dataset.iloc[0]["hour"]
                hour = hour if hour > 9 else f"0{hour}"
                minute = dataset.iloc[0]["minute"]
                minute = minute if minute > 9 else f"0{minute}"
                
                # Local Storage
                if not cloudStorage:
                    filename = f"{path}/dataset_{pair}_{interval}_{interfaceHelper.candle_lookback_length}candles_{month}-{day}-{year}_{hour}-{minute}.csv"

                    dataset.to_csv(filename)
                    print(f"--- wrote (locally) file {filename}")
                    
                    return filename
                
                # GCP - Cloud Storage
                else:
                    filename_gcp = f"datasets/{pair}/{interval}/{interfaceHelper.candle_lookback_length}_candles/dataset_{pair}_{interval}_{interfaceHelper.candle_lookback_length}candles_{month}-{day}-{year}_{hour}-{minute}.csv"

                    ## For slow upload speed
                    storage.blob._DEFAULT_CHUNKSIZE = 2097152 # 1024 * 1024 B * 2 = 2 MB
                    storage.blob._MAX_MULTIPART_SIZE = 2097152 # 2 MB

                    # import pdb
                    # pdb.set_trace()
                    
                    print("Writing csv file to google cloud storage...")
                    start_time = time.time()
                    bucket.blob(filename_gcp).upload_from_string(dataset.to_csv(), 'text/csv')
                    print(f"\n\n\n--- {round((time.time() - start_time), 1)}s to write (in GCP) file {filename_gcp}")

                    return filename_gcp

    except Exception as e:
        print('--- write_dataset_csvs error ---')
        print(e)
        return False

def fetch_predictions(filename, path):
    try: 
        filename_gcp = f"gs://{bucket_name}/predictions/{path}/{filename}"
        return pd.read_csv(filename_gcp, index_col=[0])
    except Exception as e:
        print('Error: ', e)
        return False

def write_prediction_csv(predictions_df, filename, path):
    try:
        storage_client = storage.Client()
        buecket_name = os.environ["GCP_CLOUD_STORAGE_BUCKET"]
        bucket = storage_client.get_bucket(buecket_name)

        path_filename_gcp = f"predictions/{path}/{filename}"
        
        bucket.blob(path_filename_gcp).upload_from_string(predictions_df.to_csv(), 'text/csv')
        print(f"--- wrote (in GCP) file {path_filename_gcp}")

        return path_filename_gcp

    except Exception as e:
        print('--- write_dataset_csvs error ---')
        print(e)
        return False