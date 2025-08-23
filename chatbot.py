import google.generativeai as genai
import requests
import json
import getpass
import subprocess
import time
import atexit
import os

DEM_API_URL = "http://127.0.0.1:5000/predict"
CONTEXT_FILE = "context.md"

def get_dem_prediction(features: dict):
    print(f"\nLLM is calling the DEM API with features: {features}...")
    try:
        response = requests.post(DEM_API_URL, json=features)
        response.raise_for_status()
        api_data = response.json()
        print(f"DEM API responded: {api_data}")
        return api_data
    except requests.exceptions.RequestException as e:
        error_message = f"Error calling the DEM API: {e}"
        try:
            server_error = e.response.json().get('error', 'No details provided.')
            error_message += f". Server detail: {server_error}"
        except:
            pass
        print(f"Error: {error_message}")
        return {"error": error_message}

def configure_context_file():
    print("\n--- Context Setup ---")
    print("Please provide a brief description for each feature in your dataset.")
    feature_definitions = []
    while True:
        feature_name = input("Enter feature name (or press Enter to finish): ")
        if not feature_name:
            break
        feature_desc = input(f"Enter description for '{feature_name}': ")
        feature_definitions.append(f"* **{feature_name}**: {feature_desc}")
    if not feature_definitions:
        definitions_text = "* No specific feature definitions were provided."
    else:
        definitions_text = "\n".join(feature_definitions)
    CONTEXT_TEMPLATE = """
    # System Prompt for DEM AI Assistant
        ## Your Persona
        You are a helpful and knowledgeable AI assistant for the Distilled Explanation Model (DEM). Your primary goal is to help users understand the predictions of a machine learning model that they have personally trained.
        ## Model Information
        The DEM model provides three key outputs for every prediction:
        1.  baseline_prediction: A prediction from a simple, underlying linear model.
        2.  explanation_adjustment: A positive or negative value from a rule-based decision tree that corrects the baseline.
        3.  final_prediction: The sum of the baseline and the adjustment, representing the model's final output.
        ## Feature Definitions
        The model was trained on a dataset with the following features. Use these definitions to make your explanations clear and intuitive.
        {feature_definitions}
        ## Your Task
        When a user provides feature values and asks for a prediction, you must use the `get_dem_prediction` tool. When you receive the numerical results from the tool, your response to the user should be in clear, natural language. Do not just state the numbers. Explain what they mean by breaking down the final prediction into its baseline and adjustment components, using the feature definitions above to add context.
    """
    final_context = CONTEXT_TEMPLATE.format(feature_definitions=definitions_text)
    with open(CONTEXT_FILE, 'w') as f:
        f.write(final_context)
    print(f"\nContext file '{CONTEXT_FILE}' has been created successfully.")
    return final_context

def main():
    if not os.path.exists('dem_model.pkl'):
        print("Model file 'dem_model.pkl' not found.")
        print("Please train the model first by running:")
        print("python train.py --data YOUR_DATA.csv --target YOUR_TARGET_COLUMN")
        return
    if not os.path.exists(CONTEXT_FILE):
        system_prompt = configure_context_file()
    else:
        reconfigure = input(f"Context file '{CONTEXT_FILE}' found. Use it? (Y/n): ").lower()
        if reconfigure == 'n':
            system_prompt = configure_context_file()
        else:
            with open(CONTEXT_FILE, 'r') as f:
                system_prompt = f.read()
            print(f"Using existing context from '{CONTEXT_FILE}'.")
    print("\nStarting the DEM model API server in the background...")
    api_process = subprocess.Popen(["python", "-u", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    atexit.register(api_process.terminate)
    time.sleep(3)
    while True:
        try:
            api_key = getpass.getpass("Please enter your Google AI API key and press Enter: ")
            if api_key and api_key.strip():
                genai.configure(api_key=api_key)
                break
            else:
                print("API key cannot be empty. Please try again.")
        except Exception as e:
            print(f"An error occurred while configuring the API key: {e}")
            return
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        tools=[get_dem_prediction],
        system_instruction=system_prompt
    )
    chat = model.start_chat()
    print("\nDEM Chatbot is now live! It is using the context from context.md.")
    print("Type 'quit' or 'exit' to end.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Chatbot shutting down. Goodbye!")
            break
        try:
            response = chat.send_message(user_input)
            while True:
                function_calls = []
                for part in response.candidates[0].content.parts:
                    if part.function_call.name:
                        function_calls.append(part.function_call)
                if not function_calls:
                    break
                tool_responses = []
                for function_call in function_calls:
                    if function_call.name == "get_dem_prediction":
                        args = function_call.args
                        features_data = dict(args['features'])
                        tool_output = get_dem_prediction(features=features_data)
                        tool_responses.append(
                            genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name='get_dem_prediction',
                                    response=tool_output
                                )
                            )
                        )
                response = chat.send_message(tool_responses)
            print(f"\nChatbot: {response.text}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()