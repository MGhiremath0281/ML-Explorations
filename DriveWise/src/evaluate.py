import pandas as pd
import pickle
import sys

def predict(input_file):
    # Load new data
    df = pd.read_csv(input_file)

    # Load trained model
    with open("../models/model.pkl", 'rb') as f:
        model = pickle.load(f)

    # Make predictions
    predictions = model.predict(df)

    # Save predictions
    df["Predictions"] = predictions
    df.to_csv("../data/predictions.csv", index=False)
    print("✅ Predictions saved to predictions.csv!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]  # Get input file from command line
        predict(input_file)
    else:
        print("❌ Error: No input file provided. Please provide the path to the input file as a command line argument.")
