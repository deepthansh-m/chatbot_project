import os
import pandas as pd
import pickle

try:
    # Define paths
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, 'chatbot_dataset.csv')
    response_model_path = os.path.join(base_dir, 'response_model.pkl')

    # Load dataset
    data = pd.read_csv(dataset_path)

    # Create a dictionary mapping intent to a list of responses
    response_mapping = data.groupby('intent')['bot_response'].apply(list).to_dict()

    # Save the response mapping model
    with open(response_model_path, 'wb') as file:
        pickle.dump(response_mapping, file)

    print(f"Response model saved successfully to {response_model_path}")

except Exception as e:
    print(f"An error occurred: {e}")
