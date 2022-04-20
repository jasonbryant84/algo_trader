import csv, json, os
import pandas as pd

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

def write_dataset_csvs(datasets, interfaceHelper, cloudStorage):
    try:
        storage_client = storage.Client()
        buecket_name = os.environ["GCP_CLOUD_STORAGE_BUCKET"]
        bucket = storage_client.get_bucket(buecket_name)

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
                print('dataset\n', dataset)
                
                # Local Storage
                if not cloudStorage:
                    filename = f"{path}/dataset_{pair}_{interval}_{interfaceHelper.candle_lookback_length}candles_{month}-{day}-{year}_{hour}-{minute}.csv"

                    dataset.to_csv(filename)
                    print(f"---- wrote (locally) file {filename}")
                    
                    return filename
                
                # GCP - Cloud Storage
                else:
                    filename_gcp = f"datasets/{pair}/{interval}/{interfaceHelper.candle_lookback_length}_candles/dataset_{pair}_{interval}_{interfaceHelper.candle_lookback_length}candles_{month}-{day}-{year}_{hour}-{minute}.csv"
                    
                    bucket.blob(filename_gcp).upload_from_string(dataset.to_csv(), 'text/csv')
                    print(f"---- wrote (in GCP) file {filename_gcp}")

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
    # import pdb
    # pdb.set_trace()

    try:
        storage_client = storage.Client()
        buecket_name = os.environ["GCP_CLOUD_STORAGE_BUCKET"]
        bucket = storage_client.get_bucket(buecket_name)

        path_filename_gcp = f"predictions/{path}/{filename}"
        
        bucket.blob(path_filename_gcp).upload_from_string(predictions_df.to_csv(), 'text/csv')
        print(f"---- wrote (in GCP) file {path_filename_gcp}")

        return path_filename_gcp

    except Exception as e:
        print('--- write_dataset_csvs error ---')
        print(e)
        return False