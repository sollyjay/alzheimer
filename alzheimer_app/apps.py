from django.apps import AppConfig
from django.conf import settings
import tensorflow as tf 
import os


class AlzheimerAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'alzheimer_app'
    filename = 'modelvgg.h5'
    path = os.path.join(settings.MODEL, filename)    
    load_model = tf.keras.models.load_model(path)
