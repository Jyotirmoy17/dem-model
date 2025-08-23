import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, plot_tree
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

class DEM(BaseEstimator, RegressorMixin):
    """
    Distilled Explanation Model (DEM).

    This glass-box model provides interpretable predictions by combining a
    linear baseline with a simple decision tree that provides a "distilled"
    explanation for the non-linear patterns found by a complex expert model.
    """
    def __init__(self, ridge_alpha=1.0, dt_max_depth=3, xgb_params=None):
        """
        Initializes the Distilled Explanation Model.

        Args:
            ridge_alpha (float): Regularization strength for the baseline Ridge model.
            dt_max_depth (int): Maximum depth for the explanation-distilling Decision Tree.
            xgb_params (dict): Parameters for the expert XGBoost model.
                               If None, default XGBoost parameters are used.
        """
        self.ridge_alpha = ridge_alpha
        self.dt_max_depth = dt_max_depth
        self.xgb_params = xgb_params

        self.baseline_model_ = Ridge(alpha=self.ridge_alpha)

        if self.xgb_params is None:
            self.expert_model_ = XGBRegressor(random_state=42)
        else:
            self.expert_model_ = XGBRegressor(**self.xgb_params, random_state=42)

        self.explanation_model_ = DecisionTreeRegressor(max_depth=self.dt_max_depth, random_state=42)

    def fit(self, X, y):
        """
        Fits the DEM model to the training data.

        Args:
            X (array-like): Training feature data.
            y (array-like): Training target data.

        Returns:
            self: The fitted estimator.
        """
        # Step 1: Train the interpretable baseline model
        self.baseline_model_.fit(X, y)
        y_pred_baseline = self.baseline_model_.predict(X)

        # Step 2: Train the expert model to discover complex patterns
        self.expert_model_.fit(X, y)
        y_pred_expert = self.expert_model_.predict(X)

        # Step 3: Calculate the information to be explained
        explanation_residuals = y_pred_expert - y_pred_baseline

        # Step 4: Distill the explanation into a simple, rule-based model
        self.explanation_model_.fit(X, explanation_residuals)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Makes final predictions using the fitted DEM model.

        Args:
            X (array-like): Data to make predictions on.

        Returns:
            array: The final combined predicted values.
        """
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("You must fit the model before making predictions.")

        y_pred_baseline = self.baseline_model_.predict(X)
        distilled_explanation = self.explanation_model_.predict(X)

        final_prediction = y_pred_baseline + distilled_explanation
        return final_prediction

    def predict_decomposed(self, X):
        """
        Provides the decomposed predictions from each component of the model.

        Args:
            X (array-like): Data to make predictions on.

        Returns:
            dict: A dictionary containing 'baseline', 'explanation', and 'final' predictions.
        """
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("You must fit the model before making predictions.")

        y_pred_baseline = self.baseline_model_.predict(X)
        distilled_explanation = self.explanation_model_.predict(X)
        final_prediction = y_pred_baseline + distilled_explanation

        return {
            'baseline_prediction': y_pred_baseline,
            'explanation_adjustment': distilled_explanation,
            'final_prediction': final_prediction
        }

    def visualize_explanation_tree(self, feature_names=None, figsize=(20, 10), save_path=None):
        """
        Generates a plot of the internal explanation decision tree.
        """
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("You must fit the model before visualizing it.")

        plt.figure(figsize=figsize)
        
        plot_tree(
            self.explanation_model_,
            feature_names=feature_names,
            filled=True,
            rounded=True,
            fontsize=10
        )

        plt.title("Distilled Explanation Model (DEM) - Internal Decision Tree", fontsize=16)

        if save_path:
            plt.savefig(save_path)
            print(f"Explanation tree saved to {save_path}")
        else:
            plt.show()
