# Distilled Explanation Model (DEM)

This repository contains the code for the `DEM`, a hybrid "glass-box" machine learning model, and an interactive chatbot that allows users to train the model on their own data and query it using natural language.

The project is designed to address the common trade-off between model accuracy and interpretability. It creates a powerful predictive model that remains highly explainable by breaking down its predictions into a simple baseline and a set of clear, rule-based adjustments.

## How it Works: The DEM Framework

The DEM is not a single algorithm but a three-stage framework that synergizes the strengths of both simple and complex models.

1. **Baseline & Expert Models:** For any given dataset, two models are trained:
   - A simple, interpretable **Ridge Regression** model that provides a linear baseline prediction.
   - A powerful, complex **XGBoost** model that serves as an "expert" and captures intricate, non-linear patterns.

2. **Residual Calculation:** The framework calculates the difference (the residuals) between the expert model's predictions and the baseline model's predictions. These residuals represent the complex information that the simple model failed to capture.

3. **Explanation Model (Distillation):** A single, shallow **Decision Tree** is then trained to predict these residuals. This tree learns a set of simple, human-readable rules that explain *why* the baseline prediction needs to be adjusted for certain data points. It effectively "distills" the complex knowledge from the XGBoost expert into an interpretable model.

The final prediction is the sum of the linear baseline and the rule-based adjustment from this explanation tree.

## The Interactive Chatbot and the Context File

To make the model's insights accessible, this project includes a command-line chatbot powered by Google's Gemini LLM. The chatbot's ability to provide intelligent, context-aware answers is driven by the `context.md` file.

The **`context.md`** file acts as the "brain" or instruction manual for the LLM. It contains:
- **Persona:** Defines the AI's role as an expert assistant for the DEM model.
- **Model Information:** Explains the meaning of the `baseline_prediction`, `explanation_adjustment`, and `final_prediction`.
- **Feature Definitions:** A data dictionary that you, the user, provide. This is the most critical part, as it tells the LLM what your dataset's columns actually mean (e.g., `SquareFootage` is "The total living area of a house").
- **Task:** Explicit instructions for the LLM on how to use its tools and formulate its answers.

By reading this file, the chatbot transforms from a generic AI into a specialized expert on the model *you* have trained.

## Project Structure

```
├── .gitignore
├── README.md
├── app.py
├── chatbot.py
├── context.md
├── dem/
│   ├── __init__.py
│   └── model.py
├── requirements.txt
├── setup.py
└── train.py
```

## Getting Started

Follow these steps to set up the project and run the chatbot on your local machine.

### Prerequisites

- Python 3.8 or newer
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/jyotirmoy17/dem-model.git
cd dem-model
```

### 2. Create and Activate a Virtual Environment

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it (on macOS/Linux)
source venv/bin/activate

# On Windows, use:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install .
```

## Usage Workflow 

Using the application involves two main steps: training the model and then running the chatbot.

### 1. Train the Model on Your Data

- Make sure your data is in CSV format.
- Place the CSV file inside the project directory.
- Run the train.py script from your terminal, providing the path to your data file and the name of the column you want to predict (the target).

This command creates two output files: 
- `dem_model.pkl` (the trained model and scaler) 
- `explanation_tree.png` (a visualization of the decision rules).

```bash
python train.py --data <path_to_your_data.csv> --target <your_target_column_name>
```

**Example:**  
If you have a file named `house_prices.csv` and the target column is `price`, you would run:  

```bash
python train.py --data house_prices.csv --target price
```

### 2. Run the Interactive Chatbot

- With the model trained, you can now start the interactive chatbot.
- Get a free API key from Google AI Studio.

```bash
python chatbot.py
```

- **First time running:** The script will guide you through an interactive setup to create the `context.md` file, asking you for descriptions of each feature in your dataset. A demo `context.md` file is provided.
- **Every time:** The script will securely prompt you for your Google AI API key, start the model server in the background, and launch the interactive chat.

### Example Questions to Ask

Once the chatbot is running, you can ask it questions like:

- "What is the predicted price for a 2100 square foot house with 4 bedrooms, 3 bathrooms, that is 5 years old, in a good quality neighborhood (tier 3)?"
- "I'm looking at a new 1800 sq ft house. Can you give me a price estimate?" (The chatbot will ask for the missing information).
- "What is the predicted price difference between a large, old house (3000 sq ft, 40 years old) and a small, new house (1500 sq ft, 1 year old), if they both have 3 bedrooms, 2 bathrooms, and are in an average neighborhood (tier 2)?"