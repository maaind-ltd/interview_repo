
from App.ml.Datadriven_GPVAD import vad_inference
from django.test import TestCase
from django.test import Client


test_data_folder = 'tests/test_data/'

import io
import pandas as pd
import datetime

intitial_start_time = 477628576
unix_start_time = int(datetime.datetime.now().timestamp() * 1000)

c = Client()

# TODO: Implement a test the voice-acticity-detection endpoint