#!/usr/bin/env bash

python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install -r requirements.txt

