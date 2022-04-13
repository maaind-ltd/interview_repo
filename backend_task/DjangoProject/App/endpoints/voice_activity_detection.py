from App.ml.Datadriven_GPVAD import vad_inference
from django.conf import settings
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
import pandas as pd
import numpy as np
import wavio

INCOMING_WAV_SAMPLE_RATE = 44100


class VoiceActivityDetectionEndpoint(views.APIView):
    def post(self, request):

        # TODO: Return whether a passed 'raw_audio' parameter, containing base64 encoded
        # audio data, contains any speech. Assume the passed audio has a sampling rate of 44.1 kHz

        # For this, transform the data as you need, generate a temporary .wav file with wavio.write
        # and then use
        # voice_recording_contains_speech, proportion_speech = vad_inference.get_vad_prediction(wav_audio_filepath)

        return Response("OK", status=status.HTTP_200_OK)
