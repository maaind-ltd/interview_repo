#!/usr/bin/env bash

service rsyslog restart
sleep 5

# some prints don't come through right away without v
export PYTHONUNBUFFERED=TRUE
cd /home/ubuntu/aurora_backend
source .venv/bin/activate
cd DjangoProject
gunicorn --config ../gunicorn_conf_local.py DjangoProject.wsgi
