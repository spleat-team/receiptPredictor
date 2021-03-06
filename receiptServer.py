from flask import Flask, render_template, request
import tensorflow as tf
import numpy
import cv2
from flask import jsonify
from flask_cors import CORS

from recognize_receipt.receipt_items_extractor import extract_items
from recognize_receipt.text_detector import detect_price_detector, detect_dishes_detector

# use Flask to run easily python server
app = Flask(__name__)
cors = CORS(app)

def init():
    global is_receipt_model, detect_receipt_model, digits_model, graph

    # load the model
    is_receipt_model = tf.keras.models.load_model(
        r'resources/TrainModel/classify_model.h5')

    detect_receipt_model = tf.keras.models.load_model(
        r'resources/TrainModel/detection_model.h5')

    digits_model = tf.keras.models.load_model(
        r'resources/TrainModel/digits_model.h5')

    # load the graph and use it in prediction level
    graph = tf.get_default_graph()

@app.route('/isReceipt', methods=['POST'])
def isReceipt():
    # read image file string data
    filestr = request.files['photo'].read()

    # convert string data to numpy array
    npimg = numpy.fromstring(filestr, numpy.uint8)

    return is_receipt_in_photo(npimg)

def is_receipt_in_photo(npimg):
    # convert numpy array to image and change the image size
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    smaller = cv2.resize(img, (200, 200))
    stacked_img = numpy.stack((smaller,) * 3, axis=2)
    imageToPredict = numpy.expand_dims(stacked_img, axis=0)

    # predict the model, we must use the graph unless 'model.predict' will throw an error
    with graph.as_default():
        classes = is_receipt_model.predict(imageToPredict)
    # get the result
    finalResult = numpy.argmax(classes, axis=1)[0]
    return (str(finalResult))

@app.route('/detectReceipt', methods=['POST'])
def detect_receipt():
    # read image file string data
    filestr = request.files['photo'].read()

    # convert string data to numpy array
    npimg = numpy.fromstring(filestr, numpy.uint8)

    is_receipt = is_receipt_in_photo(npimg)

    if (is_receipt == "0"):
        return False

    points1 = detect_receipt_points(npimg)
    return str(points1.tolist())

def detect_receipt_points(npimg):
    original_img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    height, width, channels = original_img.shape
    img = cv2.resize(original_img, (224, 224))
    img = img / 255.0
    image_to_predict = numpy.expand_dims(img, axis=0)
    with graph.as_default():
        pts = detect_receipt_model.predict(image_to_predict)
    pts = numpy.reshape(pts, (2, 4))
    X1 = (int(pts[0][0]), int(pts[1][0]))
    X2 = (int(pts[0][1]), int(pts[1][1]))
    X3 = (int(pts[0][2]), int(pts[1][2]))
    X4 = (int(pts[0][3]), int(pts[1][3]))
    ratio_x = width / 224.0;
    ratio_y = height / 224.0;
    points1 = numpy.float32([X1, X2, X3, X4])
    ratio = numpy.array([ratio_x, ratio_y])
    points1 = numpy.multiply(points1, ratio);
    points1 = numpy.array(points1, numpy.float32)
    return points1

@app.route('/extractReceiptItems', methods=['POST'])
def extract_receipt_items():
    # read image file string data
    filestr = request.files['photo'].read()

    # convert string data to numpy array
    npimg = numpy.fromstring(filestr, numpy.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    stacked_img = numpy.stack((img,) * 3, axis=2)

    prices = detect_price_detector(stacked_img);
    dishes = detect_dishes_detector(stacked_img);

    dishes_prices_json = extract_items(stacked_img, dishes, prices, digits_model, graph)
    return jsonify(dishes_prices_json)

@app.route('/')
def index():
    return render_template('index.html')

def main():
    # if we wnat to change a port. default 5000
    app.run(host='0.0.0.0', port=8888)
    # app.run()

init()
main()

# todo >> think if we should export isReceipt function ?
