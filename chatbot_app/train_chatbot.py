import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

try:
    # Define paths
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, 'chatbot_dataset.csv')
    intent_model_path = os.path.join(base_dir, 'intent_model.pkl')

    # Load dataset
    data = pd.read_csv(dataset_path)

    # Features and labels for intent classification
    X = data['user_input']
    y = data['intent']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with TF-IDF Vectorizer and Random Forest Classifier
    pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))

    # Train the intent classification model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print("Intent Classification Metrics:")
    print(classification_report(y_test, y_pred))

    # Save the intent classification model
    with open(intent_model_path, 'wb') as file:
        pickle.dump(pipeline, file)

    print(f"Intent model saved successfully to {intent_model_path}")

except Exception as e:
    print(f"An error occurred: {e}")
