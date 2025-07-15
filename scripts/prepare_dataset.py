from collections import namedtuple
from sys import intern
from scripts.utilities import hclip
import numpy as np

class PrepareDataset:
    def calculate_half_life(self):
        self.df['half_life'] = hclip(-self.df['delta']/np.log2(self.df['p_recall']))

    # def transform_variables(self, col):
    #     self.df['history_wrong'] = self.df['history_seen'] - self.df['history_correct']

    def create_instances_from_dataframe(self):
        Instance = namedtuple('Instance', 'p_recall delta fv half_life'.split())

        instances = []
        for _, row in self.df.iterrows():
            # Build the feature vector for this row
            fv = []
            # fv.append((intern('history_seen'), np.sqrt(1 + row['history_seen'])))
            fv.append((intern('history_correct'), np.sqrt(1 + row['history_correct'])))
            fv.append((intern('history_wrong'), np.sqrt(1 + row['history_wrong'])))
            # fv.append((intern('h_recall'), np.sqrt(1+row['h_recall'])))
            fv.append((intern('word_len'), row['word_len']))
            fv.append((intern('lang_comb:' + row['lang_combination']), 1.0))
            fv.append((intern('avg_delta'), row['avg_delta']))
            # fv.append((intern('SUBTLEX'), row['SUBTLEX']))
            fv.append((intern('std_delta'), row['std_delta']))
            fv.append((intern('avg_h_recall'), row['avg_h_recall']))
            fv.append((intern('tags_list:' + row['tags_list']), 1.0))

            instance = Instance(
                p_recall=row['p_recall'],
                delta=row['delta'],
                fv=fv,
                half_life=row['half_life']
            )

            instances.append(instance)
            
        splitpoint = int(0.8 * len(instances))
        return instances[:splitpoint], instances[splitpoint:]