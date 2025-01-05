import os
import pickle
import json
import random
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Load models
base_dir = os.path.dirname(__file__)
intent_model_path = os.path.join(base_dir, 'intent_model.pkl')
response_model_path = os.path.join(base_dir, 'response_model.pkl')

# Load the intent prediction model
with open(intent_model_path, 'rb') as file:
    intent_model = pickle.load(file)

# Load the response mapping (intent -> list of responses)
with open(response_model_path, 'rb') as file:
    response_mapping = pickle.load(file)

@csrf_exempt
def chatbot_response(request):
    if request.method == "POST":
        try:
            # Parse the user input from the request body
            data = json.loads(request.body)
            user_input = data.get('user_input', '').strip()

            # Step 1: Predict the user's intent
            predicted_intent = intent_model.predict([user_input])[0]

            # Step 2: Retrieve a random response based on the predicted intent
            responses = response_mapping.get(predicted_intent, [])
            if responses:
                response = random.choice(responses)  # Select a random response
            else:
                response = "I'm sorry, I don't understand that."

            return JsonResponse({"intent": predicted_intent, "response": response})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"message": "Send a POST request with user_input"})
