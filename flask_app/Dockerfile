FROM tensorflow/tensorflow:2.0.0-gpu-py3
WORKDIR usr/src/flask_app
COPY requirements_original.txt .
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install --no-cache-dir -r requirements_original.txt
COPY . .
