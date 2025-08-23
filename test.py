#Scrpit to test this

import numpy as np
from dem import DEM
from sklearn.model_selection import train_test_split

# --- 1. Data Preparation ---
feature_names = [f'feature_{i}' for i in range(5)]
X = np.random.rand(200, 5) * 10
y = 5 * X[:, 0] + np.cos(X[:, 1] * 2) + np.random.randn(200)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Model Training ---
model = DEM(ridge_alpha=0.5, dt_max_depth=3)
model.fit(X_train, y_train)

# --- 3. Making Predictions ---
final_predictions = model.predict(X_test)
print(f"Final predictions (first 5): {final_predictions[:5]}")

# --- 4. Inspecting the Model ---
decomposed_preds = model.predict_decomposed(X_test)
print("\n--- Decomposed Prediction for the first data point ---")
print(f"Baseline (Linear) Prediction: {decomposed_preds['baseline_prediction'][0]:.2f}")
print(f"Explanation (Tree) Adjustment: {decomposed_preds['explanation_adjustment'][0]:.2f}")
print(f"Final Combined Prediction: {decomposed_preds['final_prediction'][0]:.2f}")

# --- 5. VISUALIZATION ---
model.visualize_explanation_tree(feature_names=feature_names, save_path="explanation_tree.png")