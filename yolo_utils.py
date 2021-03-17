
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
    #from tensorflow import ConfigProto
    #from tensorflow import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import cv2
import math

class KeyPointPair:
    def __init__(self, x0, y0, Ix, Iy, c = None , classes = None):
        self.x0 = x0
        self.y0 = y0
        self.Ix = Ix
        self.Iy = Iy

        self.c     = c
        self.classes   = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.c # self.classes[self.get_label()]

        return self.score

def draw_kpp(image, g_wdt, g_hgt, kpps, labels,cls_threshold=0.8):
    image_h, image_w, _ = image.shape

    for kpp in kpps:

        x0 = int(kpp.x0)
        y0 = int(kpp.y0)
        #print('kppiy', kpp.Iy)
        #alpha = float((0.5 *math.atan2((kpp.Iy-1)/100, 1 - (kpp.Ix)/100)))
        #iy = float(1 - kpp.Iy)
        #rad = float(0.5*(math.asin(iy)))
        #print('alpha degs....',alpha,'.....rad sininv degs....', rad)
        
        #if (alpha >0) and (alpha < 91) and (rad > 0)and (rad < 1.57)
            #alpha = float(alpha - (math.pi)/2)
            #alpha = float(alpha + 1.57079633) # 70 degrees
        #else:
            #alpha = float(alpha + (math.pi)/2)
            #alpha = float(alpha - 1.57079633)

        # alpha = (kpp.alpha_norm - 0.1)/0.8*math.pi
        #alpha = kpp.alpha_norm
        alpha = float(math.atan2(kpp.Ix*2, 2*kpp.Iy - 1))
        x1 = int(kpp.x0 + (math.cos(alpha)*20))
        y1 = int(kpp.y0 + (math.sin(alpha)*20))

        #x1 = int(kpp.x0 + (math.cos(alpha)*20))
        #y1 = int(kpp.y0 + (math.sin(alpha)*20))

        cv2.circle(image, (x0,y0), 4, (0,0,127), 1)
        cv2.line( image, (x0,y0), (x1,y1), (0,0,127), 1 )

    return image

def decode_netout(netout, img_w, img_h, nb_class, obj_threshold=0.70, nms_threshold=0.25):
    grid_h, grid_w, nb_kpp = netout.shape[:3]

    kpps = []
   

    # decode output from network
    #netout[..., :2]  = _sigmoid(netout[..., :2])
    #netout[..., 2]  = _sigmoid(netout[..., 2])
    #netout[..., 2]  = _sigmoid(netout[..., 2])
    #netout[..., 2:4] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 4:])
    #netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for ikpp in range(nb_kpp):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, ikpp, 5:]

                #if np.sum(classes)>0:

                conf = netout[row,col,ikpp,4]


                #if (conf >= obj_threshold):
                #print( "conf=", conf )


                x0, y0, Ix, Iy = netout[row,col,ikpp,:4]

                x0 = ((col + _sigmoid(x0)) / grid_w) * img_w
                y0 = ((row + _sigmoid(y0)) / grid_h) * img_h

                kpp = KeyPointPair(x0, y0, Ix, Iy, conf , classes)

                kpps.append(kpp)

    # remove the kepoints which are less likely than a obj_threshold
    kpps = [kpp for kpp in kpps if kpp.get_score() > obj_threshold]

    return kpps


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x/np.min(x)*t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)
