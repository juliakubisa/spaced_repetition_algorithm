import pandas as pd 
import scipy.stats as stats
import numpy as np
from scripts.file_storage import FileStorage

class RawDataProcessor:
    def delete_outliers(self, threshold_z):
        df_outliers = self.df[['delta', 'history_seen', 'history_correct']]
        z = np.abs(stats.zscore(df_outliers))   
        self.df = self.df[(z < threshold_z).all(axis=1)]
    
    def process_data(self, input_file, input_folder, outliers_zscore_threshold, output_file, output_folder):
        self.df = FileStorage.read_data(input_file, input_folder)
        self.df.drop_duplicates(inplace=True)
        self.delete_outliers(outliers_zscore_threshold)
        FileStorage.save_data(output_file, output_folder, self.df)
