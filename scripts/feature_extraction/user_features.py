import pandas as pd
import numpy as np
from scripts.file_storage import FileStorage

class UserFeatures:
    """
    Hypothesis 1: 
    Do user features influence the probability?
    """
    def __init__(self, df):
        self.df = df

    def generate_user_features(self): 
        self.df['h_recall'] = self.df['history_correct']/self.df['history_seen']
        self.df['lang_combination'] = self.df['ui_language'] + '-' + self.df['learning_language']

        # Average p_recall specific for each user
        # self.df['avg_user_p_recall'] = self.df.groupby(['user_id', 'lang_combination'])['p_recall'].transform('mean')

        # Average interval between session 
        self.df['avg_delta'] = self.df.groupby(['user_id', 'lang_combination'])['delta'].transform('mean') 

        # Standard deviation of intervals between sessions 
        self.df['std_delta'] = self.df.groupby(['user_id', 'lang_combination'])['delta'].transform('std')

        # Average historicall recall 
        self.df['avg_h_recall'] = self.df.groupby(['user_id', 'lang_combination'])['h_recall'].transform('mean')

        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
        self.df.drop(columns=['p_recall', 'timestamp', 'delta', 'lexeme_id', 'history_seen',
                       'history_correct', 'session_seen', 'session_correct', 'ui_language', 'learning_language'], inplace=True)
        
        FileStorage.save_data('user_features.csv', 'features', self.df)