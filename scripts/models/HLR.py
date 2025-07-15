import math
import numpy as np
from collections import defaultdict, namedtuple
from scripts.utilities import pclip, hclip, mae, mean
from sklearn.metrics import r2_score
from sys import intern
import pandas as pd
from collections import defaultdict
import random
import math

class HalfLifeRegression:
    def __init__(self, learning_rate=0.001, hlwt=0.01, l2wt=0.1, sigma=1., max_half_life = 274.0, initial_weights=None):
        self.weights = defaultdict(float)  # Feature weights
        self.fcounts = defaultdict(int)    # Feature counts for adaptive learning rates
        self.learning_rate = learning_rate # Base learning rate
        self.hlwt = hlwt                   # Weight for half-life loss
        self.l2wt = l2wt                   # L2 regularization weight
        self.sigma = sigma                 # Sigma value for L2 regularization
        if initial_weights is not None:
            self.weights.update(initial_weights)
        self.max_half_life = max_half_life

    def halflife(self, inst):
        """Compute predicted half-life based on feature vector."""
        try:
            dp = sum([self.weights[k] * x_k for (k, x_k) in inst.fv])
            dp = np.clip(dp, -50, 50)
            with np.errstate(over='raise'):
                return hclip(np.exp2(dp), 15.0 / (24 * 60), self.max_half_life)
        except:
            return self.max_half_life  # Return a default max value if an error occurs

    def predict(self, inst):
        """Predict recall probability and half-life."""
        h_pred = self.halflife(inst)
        p_pred = 2 ** (-inst.delta / h_pred)  

        return pclip(p_pred), h_pred  

    
    def train_update(self, inst):
        """Update weights using one training instance."""
        p_pred, h_pred = self.predict(inst)

        # Compute gradients
        dlp_dw = 2 * (p_pred - inst.p_recall) * (math.log(2) ** 2) * p_pred * (inst.delta / h_pred)
        dlh_dw = 2 * (h_pred - inst.half_life) * math.log(2) * h_pred

        # Update weights
        for (k, x_k) in inst.fv:
            rate = (1. / (1 + inst.p_recall)) * self.learning_rate / math.sqrt(1 + self.fcounts[k])
            # Update for recall probability loss
            self.weights[k] -= rate * dlp_dw * x_k  
            
            # Update forh half-life loss 
            self.weights[k] -= rate * self.hlwt * dlh_dw * x_k  

            # L2 regularization
            self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma**2  
            self.fcounts[k] += 1


    def train(self, trainset):
        random.shuffle(trainset)
        for inst in trainset:
            self.train_update(inst)

    def losses(self, inst):
        p_pred, h_pred = self.predict(inst)
        slp = (inst.p_recall - p_pred)**2
        slh = (inst.half_life - h_pred)**2
        return slp, slh, p_pred, h_pred

    def evaluate(self, testset):
        """Evaluate the model on a test dataset."""
        results = {'p': [], 'h': [], 'pp': [], 'hh': [], 'slp': [], 'slh': []}
        for inst in testset:
            p_recall_loss, half_life_loss, p_pred, h_pred = self.losses(inst)
            results['p_recall'].append(inst.p_recall)
            results['half_life'].append(inst.half_life)
            results['p_recall_pred'].append(p_pred)
            results['half_life_pred'].append(h_pred)
            results['p_recall_loss'].append(p_recall_loss)
            results['half_life_loss'].append(half_life_loss)
            results['feature_vector'].append(inst.fv) 
            results['delta'].append(inst.delta) 

        mae_p = mae(results['p_recall'], results['p_recall_pred'])
        mae_h = mae(results['half_life'], results['half_life_pred'])
        total_slp = sum(results['p_recall_loss'])
        total_slh = sum(results['half_life_loss'])
        total_l2 = sum([x ** 2 for x in self.weights.values()])
        total_loss = total_slp + self.hlwt * total_slh + self.l2wt * total_l2
        r2val = r2_score(results['p'], results['pp'])
        print(f" MAE_P: {mae_p}, MAE_H: {mae_h}, R2: {r2val}")
        return pd.DataFrame.from_dict(results) 
    