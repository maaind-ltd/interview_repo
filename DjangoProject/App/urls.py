# flake8: noqa

import os
from django.conf import settings
from django.urls import re_path
from DjangoProject.App.endpoints.voice_activity_detection import VoiceActivityDetectionEndpoint


import warnings


warnings.filterwarnings('ignore')


# Django requires to have an app_name provided
app_name = 'Backend Test'


urlpatterns = [
    # TODO: Add your endpoint here
]
