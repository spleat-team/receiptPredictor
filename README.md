# Receipt Detector Project

## Requirement
1. Python **3.6**
2. flask installed

## Install all pip dependencies

install all pip dependencies from `requirements.txt` with:

``` bash
$ pip install -r requirements.txt
```

## How to run flask app?

on CMD:

  `$ set FLASK_APP=receiptPredictor.py`
  
  `$ flask run`

on Linux:

  `$ export FLASK_APP=receiptPredictor.py`
  
  `$ flask run`
  
default port: 5000

## Known Issues

If you encounter the following problem :
![image](https://user-images.githubusercontent.com/11838026/55881450-5c71ae00-5bab-11e9-923f-c310a3628531.png)

do : 
```
python -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
```
