from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.generics import ListAPIView
from .models import MLModel
from .files.process import make

class PrintModel(ListAPIView):
    queryset = MLModel.objects.all()

def printModel(request):
    data = make()

    print(type(data))

    return HttpResponse(status=200)
