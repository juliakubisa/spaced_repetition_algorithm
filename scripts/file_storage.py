import os
import pandas as pd

class FileStorage:
    def __init__(self):
        self.current_dir = os.getcw()

    def read_data(self, filename, folder, sep=',', names=None):
        filepath = os.path.normpath(os.path.join(self.current_dir, f'../data/{folder}/', filename))

        chunk_size = 10000
        chunks = []

        for chunk in pd.read_csv(filepath, chunksize=chunk_size, sep=sep, names=names, header=None if names else 0):
            chunk.drop_duplicates(inplace=True)
            chunk.dropna(inplace=True)
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True) 

        return df
    
    def read_subtlex_files(self, folder):
        subtlex_path = os.path.normpath(os.path.join(self.current_dir, f'../data/{folder}'))

        dfs = []
        for filename in os.listdir(subtlex_path): 
            if filename.endswith(".txt"):
                language = os.path.splitext(filename)[0].split('_')[-2]
                filepath = os.path.join(subtlex_path, filename)
                df = pd.read_csv(filepath, on_bad_lines = 'skip', sep=' ', names=['word', 'SUBTLEX'])
                df["learning_language"] = language
                dfs.append(df)
        subtlex_df = pd.concat(dfs, ignore_index=True)

        return subtlex_df
    
    def save_data(self, filename, folder, df):
        filepath = os.path.normpath(os.path.join(self.current_dir, f'../data/{folder}/'))
        df.to_csv(os.path.join(filepath, filename), sep=',', index=False, header=True)