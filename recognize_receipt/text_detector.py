import collections
import os

import cv2
from .east.model import model
import numpy as np
import tensorflow as tf
from .east.eval import detect as detect_east
from .east.eval import resize_image, sort_poly

### EAST intialization
global checkpoint_path_price_detector
global checkpoint_path_dishes_detector
checkpoint_path_price_detector = '/app/recognize_receipt/east/model_weights/price-detector/'
checkpoint_path_dishes_detector = '/app/recognize_receipt/east/model_weights/dishes-detector/'

tf.app.flags.DEFINE_string('w', '', 'gunicorn')
tf.app.flags.DEFINE_string('b', '', 'gunicorn')
tf.app.flags.DEFINE_string('timeout', '', 'gunicorn')

input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

f_score, f_geometry = model(input_images, is_training=False)

variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
saver_price_detector = tf.train.Saver(variable_averages.variables_to_restore())
saver_dishes_detector = tf.train.Saver(variable_averages.variables_to_restore())
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess_price_detector = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
sess_dishes_detector = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

ckpt_state_price_detector = tf.train.get_checkpoint_state(checkpoint_path_price_detector)
ckpt_state_dishes_detector = tf.train.get_checkpoint_state(checkpoint_path_dishes_detector)

model_path_price_detector = os.path.join(checkpoint_path_price_detector, os.path.basename(ckpt_state_price_detector.model_checkpoint_path))
model_path_dishes_detector = os.path.join(checkpoint_path_dishes_detector, os.path.basename(ckpt_state_dishes_detector.model_checkpoint_path))

print('Restore from {}'.format(model_path_price_detector))
saver_price_detector.restore(sess_price_detector, model_path_price_detector)
saver_dishes_detector.restore(sess_dishes_detector, model_path_dishes_detector)

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = pts
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    rect = np.array(rect,np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def crop_IDS(img, text_lines):
    crops = []
    for t in text_lines:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        crops.append({'image': four_point_transform(img,d), 'height': t['y0']})
    return crops

def detect_price_detector(img):
    timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])
        
    im_resized, (ratio_h, ratio_w) = resize_image(img)
    
    score, geometry = sess_price_detector.run(
            [f_score, f_geometry],
            feed_dict={input_images: [im_resized[:,:,::-1]]})
    boxes, timer = detect_east(score_map=score, geo_map=geometry, timer=timer, box_thresh=0.25)
    
    if boxes is not None:
            scores = boxes[:,8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            
    text_lines = []
    if boxes is not None:
        text_lines = []
        for box, score in zip(boxes, scores):
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                continue
            tl = collections.OrderedDict(zip(
                ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                map(float, box.flatten())))
            tl['score'] = float(score)
            text_lines.append(tl)
    
    return text_lines


def detect_dishes_detector(img):
    timer = collections.OrderedDict([
        ('net', 0),
        ('restore', 0),
        ('nms', 0)
    ])

    im_resized, (ratio_h, ratio_w) = resize_image(img)

    score, geometry = sess_dishes_detector.run(
        [f_score, f_geometry],
        feed_dict={input_images: [im_resized[:, :, ::-1]]})
    boxes, timer = detect_east(score_map=score, geo_map=geometry, timer=timer, box_thresh=0.25)

    if boxes is not None:
        scores = boxes[:, 8].reshape(-1)
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    text_lines = []
    if boxes is not None:
        text_lines = []
        for box, score in zip(boxes, scores):
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            tl = collections.OrderedDict(zip(
                ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                map(float, box.flatten())))
            tl['score'] = float(score)
            text_lines.append(tl)

    return text_lines
