from django.db import models

class MLModel(models.Model):
    name = models.CharField(max_length=100)

