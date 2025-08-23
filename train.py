import pandas as pd
import joblib
import argparse
from dem import DEM
from sklearn.preprocessing import StandardScaler

def train_model(data_path, target_column):
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    if target_column not in data.columns:
        print(f"Error: Target column '{target_column}' not found in the data.")
        return

    print("Separating features (X) and target (y)...")
    X = data.drop(columns=[target_column]).select_dtypes(include=['number'])
    y = data[target_column]
    
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training the DEM model on scaled data...")
    model = DEM()
    model.fit(X_scaled, y)

    model_path = 'dem_model.pkl'
    feature_names = list(X.columns)
    artifacts = {'model': model, 'scaler': scaler, 'feature_names': feature_names}
    joblib.dump(artifacts, model_path)
    
    print(f"\nModel and scaler have been saved to {model_path}")

    image_path = 'explanation_tree.png'
    print(f"Generating visualization of the explanation tree and saving to {image_path}...")
    model.visualize_explanation_tree(
        feature_names=feature_names,
        save_path=image_path
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DEM model on your own dataset.")
    parser.add_argument("--data", required=True, help="Path to your CSV data file.")
    parser.add_argument("--target", required=True, help="Name of the target column in your CSV file.")
    args = parser.parse_args()
    train_model(data_path=args.data, target_column=args.target)