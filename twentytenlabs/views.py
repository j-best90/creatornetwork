from django.shortcuts import render
from . import models

def home(request):
    return render(request,template_name='twentytenlabs/home.html')