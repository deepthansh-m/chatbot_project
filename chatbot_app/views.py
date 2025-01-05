import os
import pickle
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Load models
base_dir = os.path.dirname(__file__)
intent_model_path = os.path.join(base_dir, 'intent_model.pkl')
response_model_path = os.path.join(base_dir, 'response_model.pkl')

with open(intent_model_path, 'rb') as file:
    intent_model = pickle.load(file)

with open(response_model_path, 'rb') as file:
    response_mapping = pickle.load(file)

@csrf_exempt
def chatbot_response(request):
    if request.method == "POST":
        try:
            # Get user input
            data = json.loads(request.body)
            user_input = data.get('user_input', '').strip()

            # Step 1: Predict intent
            predicted_intent = intent_model.predict([user_input])[0]

            # Step 2: Retrieve response based on intent
            response = response_mapping.get(predicted_intent, "I'm sorry, I don't understand that.")

            return JsonResponse({"intent": predicted_intent, "response": response})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"message": "Send a POST request with user_input"})
