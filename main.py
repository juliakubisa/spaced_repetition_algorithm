from scripts.preprocess import load_data, clean_data
from scripts.feature_engineering import compute_word_complexity, generate_user_embeddings
from scripts.models import train_model, evaluate_model

def main():
    raw_data = load_data()
    clean_df = clean_data(raw_data)
    
    features = compute_word_complexity(clean_df)
    embeddings = generate_user_embeddings(clean_df)
    
    model = train_model(features, target="p_recall")
    results = evaluate_model(model, test_data)
    print(results)

if __name__ == "__main__":
    main()