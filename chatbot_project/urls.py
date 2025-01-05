from django.contrib import admin
from django.urls import path
from chatbot_app.views import chatbot_response  # Corrected import

urlpatterns = [
    path('admin/', admin.site.urls),
    path('chat/', chatbot_response, name='chatbot_response'),
]
