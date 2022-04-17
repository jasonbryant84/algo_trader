import csv, json, os

from google.cloud import storage

os.environ['GOOGLE_APPLICAITON_CREDENTIALS'] = "credentials.json"

def fetch_model():
    try: 
        filname_gcp = f"gs://algo-trader-staging/datasets/{pair}/{interval}/{setup_features_and_labels}_candles/{filename}"
        return pd.read_csv(filname_gcp)
    except Exception as e:
        print(e)
        return False


def save_model(pair, filename_model, model, cloudStorage):
    if cloudStorage:
        path = f"gs://algo-trader-staging/models/{pair}"
        model.save(f"{path}/{filename_model}")
    else:
        path = f"./models/{pair}"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        model.save(f"./models/{pair}/{filename_model}")

def write_csvs(datasets, interfaceHelper, cloudStorage):
    # Setting credentials for Google Cloud Platform (Cloud Storage specifically)

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
                
                # GCP - Cloud Storage
                else:
                    filename_gcp = f"datasets/{pair}/{interval}/{interfaceHelper.candle_lookback_length}_candles/dataset_{pair}_{interval}_{interfaceHelper.candle_lookback_length}candles_{month}-{day}-{year}_{hour}-{minute}.csv"
                    
                    bucket.blob(filename_gcp).upload_from_string(dataset.to_csv(), 'text/csv')
                    print(f"---- wrote (in GCP) file {filename_gcp}")

                return True

    except Exception as e:
        print(e)
        return False