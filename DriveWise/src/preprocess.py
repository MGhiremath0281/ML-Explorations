import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Identify numerical and categorical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Define the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )

    # Apply the preprocessing pipeline
    df_processed = preprocessor.fit_transform(df)
    df_processed = pd.DataFrame(df_processed, columns=numerical_features + preprocessor.transformers_[1][1].get_feature_names_out(categorical_features).tolist())
    
    return df_processed

if __name__ == "__main__":
    df = load_data(r"D:\Ml Exploration\ML-Explorations\DriveWise\Data\EDA.csv")  # Load your data
    processed_df = preprocess_data(df)
    processed_df.to_csv(r"D:\Ml Exploration\ML-Explorations\DriveWise\Data\processed_train.csv", index=False)
    print("✅ Data Preprocessed & Saved!")
