import argparse
import tqdm
import requests
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from shutil import copyfileobj
import json
from batdata.data import BatteryDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

script_dir = Path(__file__).resolve().parent
data_file_path = script_dir / 'data_link.json'

with open(data_file_path, 'r') as file:
    DOWNLOAD_LINKS = json.loads(json.load(file))


DOWNLOAD_LINKS['CAMP'] = DOWNLOAD_LINKS['CAMP'][0:3]

def download_file(url, data_dir):
    filename = url.split("/")[-1]
    data_path = data_dir / filename

    if url and not data_path.exists():
        logging.info(f"Downloading {filename} from {url}...")
        try:
            with data_path.open('wb') as fp:
                copyfileobj(requests.get(url, stream=True).raw, fp)
            logging.info(f"Downloaded {filename} successfully.")
        except Exception as e:
            logging.error(f"Failed to download {filename}: {e}")


def process_file(url, data_dir):
    data_path = data_dir / Path(url).name

    try:
        dataset = BatteryDataset.from_batdata_hdf(data_path)
        data = dataset.raw_data

        # Initialize columns with NaN
        for col in ['name', 'anode', 'anode_thickness', 'anode_loading', 'anode_porosity',
                    'cathode', 'cathode_thickness', 'cathode_loading', 'cathode_porosity',
                    'nominal_capacity', 'dataset']:
            data[col] = np.nan

        cell_meta = dataset.metadata.model_dump(exclude_defaults=True)

        # Map metadata to data
        metadata_dict = {
            'name': cell_meta.get('name', np.nan),
            'anode': cell_meta.get('battery', {}).get('anode', {}).get('product', np.nan),
            'anode_thickness': cell_meta.get('battery', {}).get('anode', {}).get('thickness', np.nan),
            'anode_loading': cell_meta.get('battery', {}).get('anode', {}).get('loading', np.nan),
            'anode_porosity': cell_meta.get('battery', {}).get('anode', {}).get('porosity', np.nan),
            'cathode': cell_meta.get('battery', {}).get('cathode', {}).get('name', np.nan),
            'cathode_thickness': cell_meta.get('battery', {}).get('cathode', {}).get('thickness', np.nan),
            'cathode_loading': cell_meta.get('battery', {}).get('cathode', {}).get('loading', np.nan),
            'cathode_porosity': cell_meta.get('battery', {}).get('cathode', {}).get('porosity', np.nan),
            'nominal_capacity': cell_meta.get('battery', {}).get('nominal_capacity', np.nan),
            'dataset': cell_meta.get('dataset_name', np.nan)
        }

        # Broadcast the metadata values to all rows
        for col, value in metadata_dict.items():
            data[col] = value

        return data

    except Exception as e:
        logging.error(f"Error processing file {file}: {e}")
        return None


def main(dataset_name, data_dir):
    all_dfs = []

    if dataset_name not in DOWNLOAD_LINKS:
        logging.error(f"Dataset {dataset_name} not found in DOWNLOAD_LINKS.")
        return pd.DataFrame()

    links = DOWNLOAD_LINKS[dataset_name]

    logging.info(f"Processing dataset: {dataset_name}")

    for url in tqdm.tqdm(links):
        if url:
            download_file(url, data_dir)
            data = process_file(url, data_dir)
            if data is not None:
                all_dfs.append(data)

    # Combine all dataframes into a single dataframe
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        logging.warning("No data was processed.")
        return pd.DataFrame()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and download datasets.")
    parser.add_argument('dataset', type=str, help='Name of the dataset to process (e.g., CAMP, SEVERSON, SNL)')
    parser.add_argument('data_dir', type=str, help='Directory to save raw data')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    combined_df = main(args.dataset, data_dir)
    if not combined_df.empty:
        combined_df.to_csv(data_dir / 'combined_data.csv', index=False)
        logging.info("Combined data saved to 'combined_data.csv'")
