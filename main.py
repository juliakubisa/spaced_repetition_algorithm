from scripts.raw_data_processor import RawDataProcessor
from scripts.file_storage import FileStorage
from scripts.feature_extraction.user_features import UserFeatures
from scripts.feature_extraction.language_features import LanguageFeatures
from scripts.models.LR import LinearRegression

def main():
    RawDataProcessor().process_data('13 million Duolingo student learning traces.csv', 'raw', 3, 'df_processed.csv', 'processed_test')
    data = FileStorage().read_data('df_processed.csv', 'processed_test')

    UserFeatures(data).generate_user_features()
    LanguageFeatures(data).generate_language_features()
    
    user_data = FileStorage().read_data('user_features.csv', 'features')
    language_data = FileStorage().read_data('language_features.csv', 'features')

    linear_regression = LinearRegression(data, user_data, language_data, 0.4)
    linear_regression.prepare_model(transform_variables_method='sqrt')
    y_train_pred, y_test_pred = linear_regression.predict()
    linear_regression.evaluate(y_train_pred, y_test_pred, include_importance=False)
    linear_regression.residuals_histogram(y_test_pred)
    linear_regression.predictions_scatterplot(y_test_pred)

    # input_df = PrepareDataset.concat_datasets(processed_df, user_features, language_features)
    # trainset, testset =  PrepareDataset.create_instances_from_dataframe(input_df)

    # hlr_model = HalfLifeRegression()
    # hlr_model.train(trainset)
    # hlr_results = hlr_model.evaluate(testset)

    # print(hlr_results)

if __name__ == "__main__":
    main()
