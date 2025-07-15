
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, BatchNormalization
from keras import regularizers, losses
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as sklearn_train_test_spli
from scripts.models.model import Model

class NeuralNetwork(Model):
    def __init__(self, df_processed, df_lang_features, df_user_features, sample_frac, method):
        self.cols_to_drop =  ['timestamp', 'lexeme_id', 'word', 'user_id', 'avg_user_p_recall', 'ui_language', 'learning_language']
        self.delta_cols_to_transform = ['delta', 'std_delta', 'avg_delta']
        self.skewed_cols_to_transform = ['history_correct', 'history_seen', 'SUBTLEX']
        self.method = method

        Model.__init__(df_processed, df_lang_features, df_user_features, sample_frac)

    def configure(self, hidden_dim = 16, l2wt = 0.1, learning_rate = 0.001, epochs = 10, batch_size = 512):
        self.hidden_dim = hidden_dim
        self.l2wt = l2wt        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def min_max_scale(self):
        if self.method == 'N-HLR':
            excluded_columns = ['p_recall', 'half_life', 'delta', 'session_seen', 'session_correct']
        else:
            excluded_columns = ['p_recall', 'session_seen', 'session_correct']

        self.numeric_features = self.df.select_dtypes(exclude=['O']).columns.drop(columns=excluded_columns)
        scaler = MinMaxScaler()
        self.df[self.numeric_features] = scaler.fit_transform(self.df[self.numeric_features])
   
    
    def calculate_hlr_loss(self):
        h_true, p_true = y_test[0], y_test[1]
        h_pred, p_pred = y_pred[0], y_pred[1]

        half_life_loss = tf.reduce_mean(tf.square(h_true - h_pred)) 
        p_recall_loss = tf.reduce_mean(tf.square(p_true - p_pred)) 

        return p_recall_loss + half_life_loss 
    
    def prepare_neural_network_architecture(self):
        self.numerical_input = Input(shape=(len(self.numeric_features),)) 
        self.delta_input = Input(shape=(1,))

        x = Dense(self.hidden_dim, activation="relu", kernel_regularizer=regularizers.l2(self.l2wt))(self.numerical_input)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        if self.method == 'N-HLR':
            half_life_output = Dense(1, activation="relu", name="half_life")(x) 
            p_recall_output = Lambda(lambda inputs: tf.pow(2.0, -inputs[0] / (inputs[1] + 1e-6)), 
                                    name="p_recall")([self.delta_input, half_life_output])
            return half_life_output, p_recall_output
    
        else:
            p_recall_output = Dense(1, activation="relu", name="p_recall")(x)
            return p_recall_output


    def predict(self):
        output = self.prepare_neural_network_architecture()

        if self.method == 'N-HLR':
            model = KerasModel(inputs=[self.numerical_input, self.delta_input], outputs=output)
            model.compile(loss=self.calculate_hlr_loss(), optimizer= Adam(learning_rate=self.learning_rate), metrics=['MAE', 'MAE'])
            model.fit([self.X_train_numerical, self.X_train_delta], self.y_train_list, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
            y_pred_half_life, y_test_pred = model.predict([self.X_test_numerical, self.X_test_delta])

        else: 
            model = KerasModel(inputs=self.numerical_input, outputs=output)
            model.compile(loss=self.calculate_hlr_loss(), optimizer= Adam(learning_rate=self.learning_rate), metrics=['MAE'])
            model.fit(self.X_train, self.y_train_list, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
            y_test_pred = model.predict(self.X_test)



    def evaluate(self, y_train_pred, y_test_pred, include_importance):
        super().evaluate(NeuralNetwork.__name__, y_train_pred, y_test_pred)


    def prepare_model(self, transform_variables_method):
        self.drop_features(self.cols_to_drop)
        self.transform_delta(self.delta_cols_to_transform)
        self.transform_variables(self.skewed_cols_to_transform, transform_variables_method)
        self.tags_threshold(1000)
        self.ohe()
        self.min_max_scale()

        if self.method=='N-HLR':
            self.split_dataset(['p_recall', 'half_life'])
        else:
            self.split_dataset('p_recall')
