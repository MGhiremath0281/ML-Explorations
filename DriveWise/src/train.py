import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def train_model():
    # Load processed data
    df = pd.read_csv(r"D:\Ml Exploration\ML-Explorations\DriveWise\Data\processed_train.csv")
    
    # Separate features and target
    X = df.drop('Daily Rental Price', axis=1)
    y = df['Daily Rental Price']

    # Split into training & validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    
    print(f"✅ Model Trained!")

    # Additional evaluation metrics for regression
    print("\nEvaluation Metrics:")
    print("R2 Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    # Ensure the directory exists
    model_dir = r"D:\Ml Exploration\ML-Explorations\DriveWise\Models"
    os.makedirs(model_dir, exist_ok=True)

    # Save model as pickle
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model Saved as Pickle!")

if __name__ == "__main__":
    train_model()
