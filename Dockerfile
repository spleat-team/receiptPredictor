FROM tensorflow/tensorflow:latest-py3
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get -y update
RUN apt-get install -y python3.6
RUN apt-get install wget
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py
RUN apt-get -y install libgeos-dev
RUN apt-get install -y libxrender-dev
RUN apt install -y libsm6 libxext6
RUN apt install -y tesseract-ocr
RUN apt install -y libtesseract-dev

WORKDIR /app
ADD . /app/

RUN python3.6 -m pip install -r /app/requirements.txt

ENV FLASK_APP /app/receiptServer.py
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD python3.6 receiptServer.py
