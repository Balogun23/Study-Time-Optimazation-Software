from django.shortcuts import render

# Create your views here.
from django.shortcuts import render

def index(request):
    return render(request, 'Study_Time_Optimization/index.html')

def predict(request):
    # Handle user input and make predictions here
    # ...

    prediction = "Sample Prediction" 
    return render(request, 'ml_app/result.html', {'prediction': prediction})
0,