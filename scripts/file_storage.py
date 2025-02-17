import os
import pandas as pd

class FileStorage:
    def __init__(self):
        self.current_dir = os.getcw()

    def read_data(self, filename, folder):
        filepath = os.path.normpath(os.path.join(self.current_dir, f'../data/{folder}/', filename))

        chunk_size = 10000
        chunks = []

        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            chunk.drop_duplicates(inplace=True)
            chunk.dropna(inplace=True)
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True) 

        return df
    
    def save_data(self, filename, folder, df):
        filepath = os.path.normpath(os.path.join(self.current_dir, f'../data/{folder}/'))
        df.to_csv(os.path.join(filepath, filename), sep=',', index=False, header=True)