from scripts.raw_data_processor import RawDataProcessor
from scripts.file_storage import FileStorage
from scripts.feature_extraction.user_features import UserFeatures
from scripts.feature_extraction.language_features import LanguageFeatures
from scripts.models.HLR import HalfLifeRegression
from scripts.models.LR import LinearRegression
from scripts.prepare_dataset import PrepareDataset

def main():
    
    RawDataProcessor.process_data('13 million Duolingo student learning traces.csv', 'raw', 
                                  'df_processed.csv', 'processed' )
    data = FileStorage.read_data('df_processed.csv', 'processed')

    UserFeatures.generate_user_features(data)
    LanguageFeatures.generate_language_features(data)

    input_df = PrepareDataset.concat_datasets(processed_df, user_features, language_features)
    trainset, testset =  PrepareDataset.create_instances_from_dataframe(input_df)

    hlr_model = HalfLifeRegression()
    hlr_model.train(trainset)
    hlr_results = hlr_model.evaluate(testset)

    lr_model = LinearRegression()
    lr_model.train()
    lr_results = lr_model.evaluate

    print(hlr_results)

if __name__ == "__main__":
    main()
