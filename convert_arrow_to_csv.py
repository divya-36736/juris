
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

# Replace with the path to your Arrow file
arrow_file_path = r'g:\My Drive\juris ai database\bhavyagiri___in_legal-sbert-dataset\default\0.0.0\1c91678dfa9dbd02e0df541d0445823cdabad2ef\in_legal-sbert-dataset-train-00000-of-00002.arrow'

# Load Arrow file into a PyArrow Table
table = pa.ipc.open_file(arrow_file_path).read_all()

# Convert PyArrow Table to a Pandas DataFrame
df = table.to_pandas()

# Save the DataFrame as a CSV file
df.to_csv('output.csv', index=False)





 








