from scripts.raw_data_processor import LoadDataset
from scripts.file_storage import FileStorage
import pandas as pd
import numpy as np
import os
import re

class LanguageFeatures:
    """
    Hypothesis 2:
    Decomposing lexeme tags into generic components 
    (e.g., part of speech, tense, gender, case) will improve the models 
    performance by reducing sparsity and better capturing word difficulty.

    Hypothesis 3:
    Introducing word complexity will influence the probability
    """ 
    def __init__(self, df):
        self.df = df[['lexeme_id', 'lexeme_string', 'learning_language']].drop_duplicates(inplace=True)
        self.lexeme_df = FileStorage.read_data('lexeme_reference.csv', 'resources', sep = ';', names=['tag', 'type', 'description'])
        self.subtlex_df = FileStorage.read_subtlex_files('resources/SUBTLEX')


        # self.df['tags'] = self.df['lexeme_string'].apply(self.extract_from_lexemestring)
        # self.df[['gender', 'POS', 'def', 'tense', 'person', 'number']] = self.df['tags'].apply(self.assign_tags)

    
    def prepare_tags_reference(self): 
        self.lexeme_df = self.lexeme_df[self.lexeme_df["type"].str.contains("adjective|animacy|other|propernoun|case") == False]
        tags_reference_dict = self.lexeme_df.set_index('tag')['type'].to_dict()
        tags_reference_types = set(tags_reference_dict.values())
        return tags_reference_dict, tags_reference_types
    

    def initialize_lexeme_types_columns(self, types):
        for lexeme_type in types:
            self.df[lexeme_type] = None

    def extract_tags_from_lexemestring(lexeme_string):
        tags = re.findall(r'<(.*?)>', lexeme_string)
        return tags 

    def add_tags(self, tags_column, tags_reference_dict):
        values = {'gender': np.nan, 'POS': np.nan, 'def': np.nan, 'tense':np.nan, 'person':np.nan, 'number':np.nan}
        for tag in tags_column:
            col = tags_reference_dict.get(tag)
            if col and pd.isna(values[col]):  # Only assign if column is empty 
                values[col] = tag
        return pd.Series([values['gender'], values['POS'],  values['def'], values['tense'], values['person'], values['number']])


    def remove_sf(self):
        """
        Some of the words contain "<sf>" at the begginig, there is no reference to it in lexeme_reference 
        """
        is_sf =  self.df['lexeme_string'].str.contains("<*sf>")
        df_sf = self.df[is_sf]
        df_without_sf = self.df[~is_sf]
        df_without_sf['word'] = df_without_sf['lexeme_string'].str.split("/").str[0]
        df_sf['word'] = df_sf['lexeme_string'].str.split("/").str[1].str.split("<").str[0]
        self.df = pd.concat([df_without_sf, df_sf])


    def add_word_len(self):
        self.df['word_len'] = self.df['word'].apply(lambda x: len(x))

    def add_subtlex(self):
        self.df = self.df.merge(self.subtlex_df, on = ['word', 'learning_language'], how='left')
        self.df.drop(columns=['lexeme_string', 'learning_language', 'word'], inplace=True)

    def generate_language_features(self):
        tags_reference_dict, tags_reference_types = self.prepare_tags_reference()
        self.initialize_lexeme_types_columns(tags_reference_types)
        self.df['tags'] = self.df['lexeme_string'].apply(self.extract_tags_from_lexemestring)
        self.df[['gender', 'POS', 'def', 'tense', 'person', 'number']] = self.df['tags'].apply(self.add_tags, 
                                                                                               args=(tags_reference_dict))
        self.remove_sf()
        self.add_word_len()
        self.add_subtlex()
        FileStorage.save_data('language_features.csv', 'features', self.df)