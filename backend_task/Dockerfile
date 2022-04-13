FROM ubuntu:18.04

RUN apt-get update && apt-get install python3.8 python3.8-dev python3.8-distutils python3.8-venv libsndfile1 -y

RUN mkdir -p /home/ubuntu/backend_example && cd /home/ubuntu

ADD ./requirements.txt /home/ubuntu/backend_example/

ADD ./setup_backend_environment.sh /home/ubuntu/backend_example/

RUN cd /home/ubuntu/backend_example/ && ./setup_backend_environment.sh

ADD ./DjangoProject /home/ubuntu/backend_example/DjangoProject

ADD ./gunicorn_conf_local.py /home/ubuntu/backend_example/

ADD ./start_backend_local.sh /home/ubuntu/backend_example/

# to fix the logging error messages
RUN apt-get install rsyslog -y
# purists wouldn't run another process in a docker but w/e
# RUN service rsyslog restart

EXPOSE 6543

RUN bash -c "cd /home/ubuntu/backend_example && source .venv/bin/activate && pip install Augment==0.4 scikit-learn==0.22.2.post1"

RUN mkdir -p /home/ubuntu/efs/

CMD cd /home/ubuntu/backend_example/ && ./start_backend_local.sh