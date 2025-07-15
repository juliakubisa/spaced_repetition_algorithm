import pandas as pd
from scripts.utilities import pclip 
import numpy as np
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt 
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc

class Model():
    def __init__(self, df_processed, df_lang_features, df_user_features, sample_frac):
        self.df = pd.merge(pd.merge(df_lang_features, df_processed, on = 'lexeme_id', how='inner'), 
                      df_user_features, on = ['user_id', 'lang_combination'], how='inner').sample(frac=sample_frac)

    def configure(self):
        pass 

    def drop_features(self, cols_to_drop):
        self.df.drop(columns=cols_to_drop, inplace=True)
    
    def clip_variables(self):
        self.df['p_recall'] = pclip(self.df['p_recall'])

    def transform_delta(self, cols):
        self.df[cols] = self.df[cols]/(60*60*24)
        
    def transform_variables(self, cols, method):
        if method == 'log':
            self.df[cols] = np.log(1e-10 + self.df[cols])
        elif method == 'sqrt':
            self.df[cols] = np.sqrt(1e-10 + self.df[cols])

    def tags_threshold(self, rare_threshold):
        tag_counts = self.df['tags_list'].value_counts()
        self.df['tags_list'] = self.df['tags_list'].apply(lambda x: x if tag_counts[x] > rare_threshold else 'rare')

    def split_dataset(self, target_cols):
        X = self.df.drop(columns=target_cols)
        y = self.df[target_cols]
        X_train, X_test, y_train, y_test = sklearn_train_test_split(X,
                                                            y,
                                                            train_size=0.8,
                                                            random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test #TODO: Split for half_life

    
    def ohe(self):
        categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()

        ohe = OneHotEncoder(sparse_output=False)
        ohe_data = ohe.fit_transform(self.df[categorical_features])
        ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(categorical_features))

        self.df = pd.concat([self.df.select_dtypes(exclude='O'), ohe_df], axis=1)
        self.df.dropna(inplace=True)
    
    
    def evaluate(self, model_name, y_train_pred, y_test_pred):
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        
        print(f"{model_name}: Train R2 = {round(train_r2, 4)}, Test R2 = {round(test_r2,4)}, Train MAE = {round(train_mae,4)}, Test MAE = {round(test_mae, 4)}")

    def residuals_histogram(self, y_test_pred):
        diff = self.y_test - y_test_pred
        plt.hist(diff)

    def plot_scatterplot(self, y_test_pred):
        plt.scatter(self.y_test, y_test_pred, alpha=0.5)
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.xlabel("Actual Recall Probability")
        plt.ylabel("Predicted Recall Probability")
        plt.title("Predicted vs. Actual Recall Probability")
        plt.show()


    def plot_calibration(outcome, y_test_pred, filename):
        prob_true, prob_pred = calibration_curve(outcome, y_test_pred, n_bins=5)
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, "s-", label="Model Calibration")
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Observed Frequency")
        plt.title("Calibration Plot")
        plt.legend()
        plt.grid()
        # plt.savefig(out_filepath(filename))

    def plot_lift(df, outcome_prob, filename):
        df["decile"] = pd.qcut(outcome_prob, 10, labels=False, duplicates='drop')  # Split into 10 deciles
        lift_df = df.groupby("decile")["outcome"].agg(["mean", "count"])
        
        baseline_rate = df['outcome'].mean()  # Overall success rate
        lift_df["lift"] = lift_df["mean"] / baseline_rate  # Calculate lift
        
        plt.figure(figsize=(6, 6))
        plt.plot(lift_df.index + 1, lift_df["lift"], marker="o", linestyle="-", color="b")
        plt.axhline(y=1, color="gray", linestyle="--", label="Baseline")
        plt.xlabel("Decile (1 = Highest Predictions)")
        plt.ylabel("Lift")
        plt.title("Lift Chart")
        plt.legend()
        plt.grid()
        # plt.savefig(out_filepath(filename))