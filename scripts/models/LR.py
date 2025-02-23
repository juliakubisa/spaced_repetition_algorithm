import pandas as pd
from scripts.models.model import Model
from scripts.utilities import cap_y
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

class LinearRegression(Model):
    def __init__(self, df_processed, df_lang_features, df_user_features, sample_frac):
        self.cols_to_drop =  ['timestamp', 'lexeme_id', 'word', 'user_id', 'session_seen', 'session_correct', 
                              'avg_user_p_recall', 'ui_language', 'learning_language']
        self.delta_cols_to_transform = ['delta', 'std_delta', 'avg_delta']
        self.skewed_cols_to_transform = ['history_correct', 'history_seen', 'SUBTLEX']

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])

        Model.__init__(df_processed, df_lang_features, df_user_features, sample_frac)

    def ohe(self):
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        ohe = OneHotEncoder(sparse_output=False)
        ohe_data = ohe.fit_transform(self.df[categorical_cols])
        ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(categorical_cols))
        self.df = pd.concat([self.df.select_dtypes(exclude='O'), ohe_df], axis=1)
        self.df.dropna(inplace=True)

    def predict(self):        
        self.pipeline.fit(self.X_train, self.y_train)

        y_train_pred = cap_y(self.pipeline.predict(self.X_train))
        y_test_pred = cap_y(self.pipeline.predict(self.X_test)) 

        return y_train_pred, y_test_pred

    def evaluate(self, y_train_pred, y_test_pred, include_importance):
        super().evaluate(LinearRegression.__name__, y_train_pred, y_test_pred)

        if include_importance:
            importances_list = []  
            model = self.pipeline.named_steps["model"]
            importances_list.append((self.X_train.columns, model.coef_))

            importance_df = pd.DataFrame(importances_list, columns = ['Feature', 'Importance'])

            importance_df = importance_df.explode(["Feature", "Importance"]).reset_index(drop=True)
            tags_importance = importance_df[importance_df['Feature'].str.contains("tags_list_")].copy()

            overall_tags_importance = tags_importance['Importance'].abs().sum()

            importance_df.loc[-1] = ['Linear Regression', 'tags_list (overall)', overall_tags_importance]
            importance_df.index = importance_df.index + 1 

            importance_df['Absolute_Val'] = importance_df['Importance'].abs()
            importance_df = importance_df.sort_values(by='Absolute_Val', ascending=False)
            print(f'Importance: {importance_df[:10]}')

    def prepare_model(self, transform_variables_method):
        self.drop_features(self.cols_to_drop)
        self.transform_delta(self.delta_cols_to_transform)
        self.transform_variables(self.skewed_cols_to_transform, transform_variables_method)
        self.tags_threshold(1000)
        self.ohe()
        self.split_dataset()
