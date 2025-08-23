# A Test file to demonstrate how to run this

import unittest
import os
import subprocess
import pandas as pd
import requests
import time
from threading import Thread

TEST_DATA_PATH = 'test_data.csv'
MODEL_PATH = 'dem_model.pkl'
API_URL = "http://127.0.0.1:5000/predict"

class TestDemWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up test environment...")
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'target': [3, 5, 7, 9, 11]
        }
        df = pd.DataFrame(data)
        df.to_csv(TEST_DATA_PATH, index=False)

    @classmethod
    def tearDownClass(cls):
        print("\nTearing down test environment...")
        if os.path.exists(TEST_DATA_PATH):
            os.remove(TEST_DATA_PATH)
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

    def test_1_model_training(self):
        print("\n--- Testing train.py ---")
        command = [
            "python", "train.py",
            "--data", TEST_DATA_PATH,
            "--target", "target"
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Training script failed with error:\n{result.stderr}")
        self.assertTrue(os.path.exists(MODEL_PATH), "train.py did not create the model file.")
        print("train.py ran successfully and created dem_model.pkl")

    def test_2_api_prediction(self):
        print("\n--- Testing app.py API ---")
        if not os.path.exists(MODEL_PATH):
            self.skipTest("Skipping API test because model file does not exist.")

        from app import app
        server_thread = Thread(target=app.run, kwargs={'port': 5000})
        server_thread.daemon = True
        server_thread.start()
        time.sleep(2)

        sample_features = {'feature1': 2.5, 'feature2': 3.5}
        response = requests.post(API_URL, json=sample_features)
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn('final_prediction', response_data)
        self.assertIsInstance(response_data['final_prediction'], list)
        
        print("app.py started and served a valid prediction.")

if __name__ == '__main__':
    unittest.main()