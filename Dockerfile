FROM tensorflow/tensorflow:latest-py3

WORKDIR /app
ADD ./receiptServer.py /app
ADD ./templates/index.html /app/templates/
ADD ./resources/TrainModel/classify_model.h5 /app/resources/TrainModel/

RUN pip install FLASK
RUN pip install opencv-python-headless

ENV FLASK_APP /app/receiptServer.py
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD flask run --host=0.0.0.0 --port=8888