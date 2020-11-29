
from azureml.core import Run
from azureml.train.automl import AutoMLConfig
import pandas
import numpy as np 
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import os

# Get the experiment run context
run = Run.get_context()

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, dest='output_folder', help='output folder')
args = parser.parse_args()
output_folder = args.output_folder

# load the titanic dataset
print("Loading Data...")
train_df = run.input_datasets['Titanic'].to_pandas_dataframe()

train_df = train_df.drop(['PassengerId'], axis=1)

# Save prepared data
os.makedirs(output_folder, exist_ok=True)
pq.write_table(pa.Table.from_pandas(train_df), output_folder)
