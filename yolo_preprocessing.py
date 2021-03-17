
import os # open the folder
import cv2 # Image editing
import copy
import numpy as np # math, arrays
import imgaug as ia # adding filters like noise reduction
from imgaug import augmenters as iaa 
import imgaug as ia
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import collections
    #from tensorflow import ConfigProto
    #from tensorflow import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from keras.utils import Sequence
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt # fig plotting (we can plot in chart)
from yolo_utils import draw_kpp 
import math # square root, sin ,cos

def read_annotations(img_dir):
    all_imgs = []
    seen_labels = {}
    grid = 16
    pixel = 128
    px_grd = pixel/grid

    for ann_file_name in sorted(os.listdir(img_dir)):
        ext = os.path.splitext( ann_file_name )[1]

        if ext == '.txt':
            img = {'object':[]}

            img_file_name = img_dir + ann_file_name #adding the strings
            print("img_file_name=", img_file_name)
            print( "ann_file_name=", ann_file_name )

            # read in predefined order
            file = open( img_file_name, "r" ) # r means only to read

            for line in file:  # each line is a keypoint pair to this picture
                vWords = line.split()
                obj = {}
                x =[]

                img['filename'] = os.path.splitext( img_file_name )[0] + ".bmp"  # image file name
                obj['name'] = vWords[0]  # class name

                if obj['name'] in seen_labels:
                    seen_labels[obj['name']] += 1
                else:
                    seen_labels[obj['name']] = 1

                ## two keypoints which give pose (x0, y0) and direction (x1, y1) in pixel coordinates, all following coordinates are from object polygon and thus ignored
                #if float(vWords[4])< (0.8*60/180)+0.1 or  float(vWords[4])> (0.8*120/180)+0.1:
                #if float(vWords[4])< (10*math.pi/180) or  float(vWords[4])> (170*math.pi/180):
                #    if float(vWords[5])> 0.0 and float(vWords[6])> 0.0:
                      

                            
                # alpha --> gamma in future
                obj['x0'] = float( vWords[1] )
                obj['y0'] = float( vWords[2] )
                # obj['alpha'] = math.pi -  ((float( vWords[3]) - 0.1)/0.8*math.pi)
                # obj['alpha'] = -  ((float( vWords[3]) - 0.1)/0.8*math.pi)
                obj['alpha'] = float(vWords[3])
            
                obj['Ix'] = float(2*math.sin(obj['alpha'])**2)
                #obj['Ix'] = float(np.degrees(obj['alpha']))
                
                obj['Iy'] = float(math.sin(2*obj['alpha']))
                obj['Ix'] = obj['Ix'] * 0.5
                obj['Iy'] = (obj['Iy'] + 1) * 0.5
                #print("obj alpha :",obj['alpha'],"   Ix :", obj['Ix'],"   Ixy:", obj['Iy'])
                #obj['Iy'] = float(math.degrees(obj['alpha']))
                
                #alpha = (float( vWords[3] ) - 0.1)/0.8*math.pi #alpha in rad
                
                #alpha = math.atan2(obj['Ix']**0.5, obj['Iy']**0.5)
                alpha = float(math.atan2(obj['Ix']*2,obj['Iy']*2 -1))
##                        print("Cal. alpha", alpha)
                if obj["alpha"] > math.pi:
                   alpha = alpha + math.pi

                #print("Cal. alpha ", alpha)
                x1 = (float( vWords[1] ) + math.cos( obj['alpha'] )*20)
                y1 = (float( vWords[2] ) + math.sin( obj['alpha'] )*20)
                obj['x1'] = x1
                obj['y1'] = y1
                cell_x = np.floor(obj['x0']/px_grd)+1
                cell_y = np.floor(obj['y0']/px_grd)
                obj['cell_no'] = (grid*cell_y )+ cell_x
                #obj['cell_no'] = round(obj['cell_no']/grid)
                #print(obj['cell_no'])
                obj['compoconf'] = 0.5 * float(vWords[5]) + 0.5 * float(vWords[6])

                # extend dict obj['cellX'] and obj['cellY'] and weighted sum obj['compoconf'] = w0*vWords[5] + w1*vWords[6] and obj['deleteMark']=0
                img['object'] += [obj]  #this is the keypo int pair of this image
            #print('img object original',len(img['object']))

            for i in img['object']:
                x.append(i['cell_no'])
            y = collections.Counter(x) # counting the repeatation of the cell_nos
            dups =[] # list of duplicates
            inx = [] # list of indices to be removed
            confarr =[] # list of confidences for comparison
            for k in y:
                if y[k] > 1:
                    dups.append(k) # taking the duplicates 
            for i in dups:
                for j in img['object']:
                    if j['cell_no'] == i: # taking the weighted-confidences of duplicates
                        confarr.append(j['compoconf'])
                maxtemp = np.max(confarr) # determining the maximum weighted-confidence of duplicates
                for k in img['object']:
                    if i ==k['cell_no'] and k['compoconf'] != maxtemp: # gathering the lower weighted-confidence's indices
                        inx.append(img['object'].index(k))
                        
                        
                #print(i,'confidences',confarr)
                #print(i,'dup index',inx)
                confarr =[]    
                #inx = []
            inx.sort(reverse=True)
            for l in inx:
                del img['object'][l] # removing the lower weighted confidence from the annotation
            
            #print('# of indices to be removed',len(inx))
            #print('counter cell nos',y)
            #print('dups',dups)
            #print('final annotation going',img['object'])
            #print('final annotation length',len(img['object']))        

            file.close()
            
            #
            # remove here all conflicting objects with same cell but lower weighted sum of confidences
            # loop index0 = 0 up to sizeof( img['object'] ) - 2)
            #   obj0 = img['object'][index0]
            #   loop index1 = index0+1 up to ( sizeof( img['object'] ) - 1)
            #       obj1 = img['object'][index1]
            #       if ( obj1['cellX'] == obj0['cellX'] ) && ( obj1['cellY'] == obj0['cellY'] ) && (obj1['compoconf'] < obj0['compoconf']):
            #           obj['deleteMark'] = 1
            # loop over all objects and delete marked objects from img 
            #   >>> l = [1,2,3,4,4,5,5,6,1]
            #       set([x for x in l if l.count(x) > 1])
            #       set([1, 4, 5])

            if len(img['object']) > 0: # len means number of items in the list
                all_imgs += [img]
    return all_imgs, seen_labels

class YoloBatchGenerator(Sequence):
    def __init__(self, images,
                       config,
                       shuffle=True,
                       jitter=True,
                       norm=None):
        self.generator = None # it means nothing

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm
        self.image_counter = 0

        ia.seed( 1 )


        ### augmentors by https://github.com/aleju/imgau/g
        sometimes = lambda aug: iaa.Sometimes(0.8, aug) # lambda is a single command

        # hier wird die Augmentation definiert, aber nicht ausgeführt.
        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                sometimes( iaa.Affine(
                    #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-10, 10), # rotate by -45 to +45 degrees
                    #shear=(-5, 5), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    mode = "edge"

                )),
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        #    iaa.CoarseSalt(0.01, size_percent=(0.002, 0.01)),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        #sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0), # randomly remove up to 10% of the pixels
                            #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        #iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-15, 15), per_channel=0), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ])

            ],
            random_order=True
        )

        if shuffle: np.random.shuffle(self.images)


    def __len__(self): # __ means specail mention (lenght of the object)
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['x0'], obj['y0'], obj['x1'], obj['y1'], self.config['LABELS'].index(obj['name'])]
            #annot = [obj['x0'], obj['y0'], obj['alpha'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])

    def __getitem__(self, idx):  # get a complete batch
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        best_anchor = 0
        # TRUE_KPP_BUFFER == max_kpp_per_image (here e.g. 1 in config), list of keypoints x0, y0, x1, y1 one keypoints-pair per grid cell, same as in y_batch
        # but in ascending order, box-by-box

        # <batchsize> <BOX == nb_box == len( anchors )/2>, <for each box 4*keypoint pairs +  1*confidency + one-hot labels> == desired network output tensor
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['KPP'], 4+1+len(self.config['LABELS'])))

        i_img = 0
        keypoints_on_images = [] # contains all keypoints in the image, ready for augmentation
        images_batch = []

        #create list of keypoints
        num_images = len( self.images )
        instance_src_index = l_bound % num_images
        for instance_count in range( r_bound - l_bound ):
            train_instance = self.images[instance_src_index]
            # augment input image and fix object's position and size
            image_name = train_instance['filename']
            img = cv2.imread(image_name)
            img = img[:,:,1]  # green channel only
            img = np.expand_dims( img, -1 )  # reattach a dimension

            images_batch.append( img )

            # construct output from object's x, y, w, h
            keypoints_on_image = []
            all_objs = train_instance['object']
            for obj in all_objs:
                keypoints_on_image.append( ia.Keypoint( x=float(obj['x0']), y=float(obj['y0']) ))
                keypoints_on_image.append( ia.Keypoint( x=float(obj['x1']), y=float(obj['y1']) ))
                                
            keypoints_on_images.append(ia.KeypointsOnImage(keypoints_on_image, shape=img.shape ))
            instance_src_index = (instance_src_index + 1) % num_images

        if self.jitter:
            ia.seed( 134 )
            aug_pipe_det = self.aug_pipe.to_deterministic() # so that the augmentation of the images and the keypoints effect the same transformations
            x_batch = aug_pipe_det.augment_images(images_batch) # augmented images
            keypoints_batch_aug = aug_pipe_det.augment_keypoints( keypoints_on_images )  # augmented keypoints
        else:
            x_batch = images_batch
            keypoints_batch_aug = keypoints_on_images

        x_batch = np.reshape( x_batch, (r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 1 ) )
        x_batch = self.norm( x_batch )

        # enter augmented keypoints in y_batch
        num_images = len( self.images )
        instance_src_index = l_bound % num_images
        for instance_count in range( r_bound - l_bound ):
            train_instance = self.images[instance_src_index]
            all_objs = train_instance['object']
            obj_count = 0
            for obj in all_objs:
                if obj['name'] in self.config['LABELS']:
                    kp0_x = keypoints_batch_aug[instance_count].keypoints[obj_count*2].x
                    kp0_x = kp0_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    kp0_y = keypoints_batch_aug[instance_count].keypoints[obj_count*2].y
                    kp0_y = kp0_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])
                    #alpha = float(keypoints_batch_aug[instance_count].keypoints[obj_count*2+1].x)
                    #alpha = ((alpha - 0.1)/0.8*math.pi)
                    kp1_x = keypoints_batch_aug[instance_count].keypoints[obj_count*2+1].x
                    kp1_x = kp1_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    kp1_y = keypoints_batch_aug[instance_count].keypoints[obj_count*2+1].y
                    kp1_y = kp1_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    dx = kp1_x - kp0_x
                    dy = kp1_y - kp0_y

                    #alpha = math.atan2(dx,dy)
                    
                    a = (math.atan2(dy,dx))
                    #Ix = float(np.degrees(a))
                    Ix = float(2*math.sin(a)**2)
                    Ix = Ix * 0.5
                    Iy = float(math.sin(2*a) + 1.0)
                    Iy = (Iy + 1) * 0.5
                    #print("bottom alpha :", a),#"   bottom Ix:", Ix,"   bottom Iy:", Iy ,"\n")
                    #lngth = math.sqrt( dx*dx + dy*dy )
                    #if lngth > 0.0:
                    #    ex=dx/lngth
                    #    ey=dy/lngth
                                            
                    # Determine the grid cell to which the keypoint belongs.
                    grid_x = int(np.floor(kp0_x))  #these are the grid coordinates, e.g. in the 4x4 grid into which the image is divided
                    grid_y = int(np.floor(kp0_y))



                    if grid_x >= 0 and grid_y >= 0 and grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx  = self.config['LABELS'].index(obj['name'])   #label-class-number

                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        best_anchor = 0  #vorerst nur ein anchor-keypoint je grid_cell
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0] = kp0_x  #keypoint0 in grid-Koordinaten LUC
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 1] = kp0_y
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 2] = Ix
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 3] = Iy
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.  #confidence, is always 1.0 in gound truth
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5+obj_indx] = 1.  #one-hot class

                        self.image_counter += 1

                    obj_count += 1
            instance_src_index = (instance_src_index + 1) % num_images


        return x_batch, y_batch   #image normalized and y_batch in grid coordinates

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)  #shuffle along the first axis only
