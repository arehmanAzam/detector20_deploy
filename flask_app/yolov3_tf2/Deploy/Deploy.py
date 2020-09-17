import time
from absl import logging
# from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from ..yolov3_tf2_files.models import (
    YoloV3, YoloV3Tiny
)
from ..yolov3_tf2_files.dataset import transform_images, load_tfrecord_dataset
from ..yolov3_tf2_files.utils import draw_outputs
import flask
from flask import Flask,request
from collections import OrderedDict
import io
import base64
import numpy
from imageio import imread
import os
import json
import webcolors

classes=os.getcwd()+'/yolov3_tf2/'+'data/voc2012.names'
weights=os.getcwd()+'/yolov3_tf2/'+'/checkpoints/yolov3_train_7.tf'
tiny=False,
size= 416
image= './data/girl.png'
tfrecord= None
output='./output.jpg'
num_classes= 4
class_names=[]
yolo=None
first_run_flag=True

class_mappings={
        'rayban01':'Rayban Wayfarer',
        'Oo9343':'Oakley Men\'s Oo9343 M2 Frame Xl Shield Sunglasses',
        'ck01': 'CK One Eau De Toilette',
        'oakleySun2':'Oakley Sunglasses 2'}
# flags = tf.compat.v1.flags

def initializations():
    global yolo
    global weights
    global classes
    global class_names,num_classes
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    yolo = YoloV3(classes=num_classes)

    checkpoint_dir = os.path.dirname(weights)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    logging.info("loading model")
    yolo.load_weights(weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(classes).readlines()]
    logging.info('classes loaded')
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name
def color_find(np_im,boxes,max_index):
    img = cv2.cvtColor(np_im, cv2.COLOR_RGB2BGR)
    wh = np.flip(img.shape[0:2])

    x1y1 = tuple((np.array(boxes[0][0:2]) * wh).astype(np.int32))
    x2y2 = tuple((np.array(boxes[0][2:4]) * wh).astype(np.int32))
    ROI = img[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]
    histb = cv2.calcHist([ROI[0]], [0], None, [256], [0, 256])
    histg = cv2.calcHist([ROI[1]], [0], None, [256], [0, 256])
    histr = cv2.calcHist([ROI[2]], [0], None, [256], [0, 256])
    blue_val=int(np.where(histb == np.max(histb))[0])
    green_val=int(np.where(histg == np.max(histg))[0])
    red_val=int(np.where(histr == np.max(histr))[0])
    actual,closest =get_colour_name((red_val,green_val,blue_val))
    return closest
initializations()
app = Flask(__name__)


@app.route("/", methods=["POST"])
def predictions():
    try:
        sbuf = io.StringIO()
        _json_response = OrderedDict()
        global size, class_names, yolo, classes
        if flask.request.method == "POST":
            if request.data:
                max_class_name = 'No detection'
                second_max_class_name = 'No detection'
                max_score = 0
                second_max_score = 'None'
                max_result_roi = None
                second_max_roi = None
                max_result_roi_string = 'None'
                second_max_roi_string = 'None'
                closest_color='None'
                # print(type(request.data))
                content = request.json

                logging.info("Image resolution %s"%content["Image_resolution"])
                logging.info("Bytes sent %s" %content["bytes_sent"])
                b64_string=content['Image']

                # b64_string = request.data.decode()
                logging.info("Size of Image recieved")
                logging.info((len(b64_string)*3)/4)
                # sbuf.write(base64.b64decode(request.data.decode()))
                # nparr = np.fromstring(request.data, np.uint8)
                bytes_image = io.BytesIO(base64.b64decode(b64_string))
                image1 = imread(bytes_image)
                np_im = np.array(image1)
                img_in = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                img_in = tf.expand_dims(img_in, 0)
                img_in = transform_images(img_in, size)
                t1 = time.time()
                boxes, scores, classes, nums = yolo(img_in)

                t2 = time.time()
                logging.info('time: {}'.format(t2 - t1))

                logging.info('detections:')
                counter=0
                for i in range(nums[0]):
                    logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                       np.array(scores[0][i]),
                                                       np.array(boxes[0][i])))
                    counter+=1
                proto_tensor = tf.make_tensor_proto(scores[0])
                scores=tf.make_ndarray(proto_tensor)
                proto_tensor = tf.make_tensor_proto(boxes[0])
                boxes = tf.make_ndarray(proto_tensor)
                proto_tensor = tf.make_tensor_proto(classes[0])
                classes = tf.make_ndarray(proto_tensor)
                if range(nums[0]) is not None :
                    if range(nums[0]).stop > 0:
                        max_result_roi = boxes[scores.argmax()]
                        roi=np.array(boxes[0])
                        max_score = int(scores.argmax())
                        max_class_name = class_mappings[class_names[int(classes[scores.argmax()])]]
                        max_result_roi_string = ', '.join(map(str, max_result_roi))
                        closest_color=color_find(np_im,boxes,scores.argmax())

                        # ROI = np_im[top:bottom, left:right]
                    if range(nums[0]).stop > 1:
                        second_max_score = ''
                        second_max_class_name = ''
                        second_max_roi_string = ''
                        for i in range(nums[0]):
                            if i!=0:
                                second_max_score+=str(np.array(scores[0][i]))+', '
                                second_max_class_name += class_mappings[str(class_names[int(classes[0][i])])] + ', '
                                second_max_roi_string += str(np.array(boxes[i])) + ', '
                        #Will be coded when classes get to increase
                _json_response = OrderedDict()
                _json_response['first_score'] = max_score
                _json_response['first_class'] = max_class_name
                _json_response['first_roi'] = max_result_roi_string
                _json_response['first_color']=closest_color
                _json_response['other_scores'] = second_max_score

                _json_response['other_classes'] = second_max_class_name

                _json_response['other_rois'] = second_max_roi_string
                _json_response['time_sent'] = time.strftime('%Y-%m-%d %H:%M:%S')
                _json_response['Success']=True
                _json_response['Exception']='None'
                logging.info(_json_response)
            return json.dumps(_json_response, sort_keys=True)
    except Exception as e:
        _json_response = OrderedDict()
        _json_response['first_score'] = max_score
        _json_response['first_class'] = max_class_name
        _json_response['first_roi'] = max_result_roi_string
        _json_response['other_scores'] = second_max_score

        _json_response['other_classes'] = second_max_class_name

        _json_response['other_rois'] = second_max_roi_string
        _json_response['first_color'] = closest_color
        _json_response['time_sent'] = str(time.strftime('%Y-%m-%d %H:%M:%S'))
        _json_response['Success'] = False
        _json_response['Exception'] = str(e)
        logging.info("Exception occured in flask. Exception: %s"%e)
        return json.dumps(_json_response, sort_keys=True)


if __name__ == '__main__':
    try:
        app.run('0.0.0.0')
    except SystemExit:
        pass


