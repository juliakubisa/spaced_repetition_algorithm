from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from scripts.models.utilities import cap_y
import matplotlib.pyplot as plt


data_root = 'https://autogluon.s3.amazonaws.com/datasets/Inc/'
train_data = TabularDataset(data_root + 'train.csv')
test_data = TabularDataset(data_root + 'test.csv')


label = 'p_recall'
presets = 'medium'
predictor = TabularPredictor(label=label).fit(train_data, excluded_model_types = ['NeuralNetTorch', 'NeuralNetFastAI', 'KNeighborsDist', 'KNeighborsUnif', 'LightGBM', 'LightGBMXT', 'RandomForestMSE'])predictions = predictor.predict(test_data)
predictions = predictor.predict(test_data)

y_pred = predictor.predict(test_data.drop(columns=[label]))
y_pred = cap_y(y_pred)

y_test = test_data['p_recall']
# results_df = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred, 'session_seen':test_data['session_seen'], 
#                           'session_correct':test_data['session_correct']})

# print('mae:', (results_df['y_test'] - results_df['y_pred']).abs().mean())




predictions_scatterplot = plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.xlabel("Actual Recall Probability")
plt.ylabel("Predicted Recall Probability")
plt.title("Predicted vs. Actual Recall Probability")