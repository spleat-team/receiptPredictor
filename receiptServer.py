from flask import Flask, render_template, request
import tensorflow as tf
import numpy
import cv2

# use Flask to run easily python server
app = Flask(__name__)

class server:
    def __init__(self):
        global model, graph
        filename = 'resources/TrainModel/classify_model.h5'

        # load the model
        model = tf.keras.models.load_model(filename)

        # compile the model - required?
        sgd = tf.keras.optimizers.SGD(lr=0.01, decay=0.00, momentum=0.9, nesterov=False)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        # load the graph and use it in prediction level
        graph = tf.get_default_graph()

    @app.route('/isReceipt', methods=['POST'])
    def isReceipt():
        # read image file string data
        filestr = request.files['photo'].read()

        # convert string data to numpy array
        npimg = numpy.fromstring(filestr, numpy.uint8)

        # convert numpy array to image and change the image size
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        smaller = cv2.resize(img, (200, 200))

        stacked_img = numpy.stack((smaller,)*3, axis=2)
        imageToPredict = numpy.expand_dims(stacked_img, axis=0)

        # predict the model, we must use the graph unless 'model.predict' will throw an error
        with graph.as_default():
            classes = model.predict(imageToPredict)

        # get the result
        finalResult = numpy.argmax(classes, axis=1)[0]

        return (str(finalResult))

@app.route('/')
def index():
    return render_template('index.html')

def main():
    # if we wnat to change a port. default 5000
    #app.run(host='127.0.0.1', port=8888)
    app.run()
    server()

main()

# todo >> think if we should export isReceipt function ?
