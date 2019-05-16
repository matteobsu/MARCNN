import numpy
#import threading
import multiprocessing
import random
import ctypes
from scipy import linalg
from scipy import ndimage
from keras.preprocessing.image import load_img, img_to_array, array_to_img#, save_img
from keras.utils import Sequence
import glob
#from PIL import Image
from skimage import util
from skimage import transform
import itertools
import os
import re
import h5py


WORKERS = 12
CACHE_SIZE = 32
#dims = (256,256,1)
#NORM_DATA_MODE = 0  # 0 - per image over all channels; 1 - per image per channel; 2 - flat-field norm
TYPES = {"XRAY": 0, "SCATTER": 1, "CT": 2}

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(numpy.round(h * zoom_factor))
        zw = int(numpy.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = numpy.zeros_like(img)
        out[top:top+zh, left:left+zw] = ndimage.zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(numpy.round(h / zoom_factor))
        zw = int(numpy.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def numpy_normalize(v):
    norm = numpy.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm

def get_min_max(a, numChannels, flatField=None, itype=TYPES["XRAY"], minx=None, maxx=None):
    inShape = a.shape
    inDimLen = len(inShape)
    a = numpy.squeeze(a)
    outShape = a.shape
    outDimLen = len(outShape)
    if itype == TYPES['XRAY']:
        if flatField!=None:
            a = numpy.clip(numpy.divide(a, flatField), 0.0, 1.0)
        else:
            if numChannels<=1:
                if minx is None:
                    minx = numpy.min(a)
                if maxx is None:
                    maxx = numpy.max(a)
            else:
                if minx is None:
                    minx = []
                if maxx is None:
                    maxx = []
                if outDimLen<4:
                    for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                        if len(minx)<numChannels:
                            minx.append(numpy.min(a[:, :, channelIdx]))
                        if len(maxx)<numChannels:
                            maxx.append(numpy.max(a[:, :, channelIdx]))
                else:
                    for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                        if len(minx)<numChannels:
                            minx.append(numpy.min(a[:, :, :, channelIdx]))
                        if len(maxx)<numChannels:
                            maxx.append(numpy.max(a[:, :, :, channelIdx]))
    elif itype == TYPES['SCATTER']:
        if numChannels<=1:
            if minx is None:
                minx = 0
            if maxx is None:
                maxx = numpy.max(numpy.abs(a))
        else:
            if minx is None:
                minx = []
            if maxx is None:
                maxx = []
            if outDimLen < 4:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    #minx.append(numpy.min(flatField[:, :, channelIdx]))
                    #maxx.append(numpy.max(flatField[:, :, channelIdx]))
                    if len(minx) < numChannels:
                        minx.append(0)
                    if len(maxx) < numChannels:
                        maxx.append(numpy.max(numpy.abs(a[:, :, channelIdx])))
            else:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    #minx.append(numpy.min(flatField[:, :, channelIdx]))
                    #maxx.append(numpy.max(flatField[:, :, channelIdx]))
                    if len(minx) < numChannels:
                        minx.append(0)
                    if len(maxx) < numChannels:
                        maxx.append(numpy.max(numpy.abs(a[:, :, :, channelIdx])))
    elif itype == TYPES['CT']:
        if numChannels<=1:
            if minx is None:
                minx = numpy.min(a)
            if maxx is None:
                maxx = numpy.max(a)
        else:
            if minx is None:
                minx = []
            if maxx is None:
                maxx = []
            if outDimLen < 4:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    if len(minx) < numChannels:
                        minx.append(numpy.min(a[:, :, channelIdx]))
                    if len(maxx) < numChannels:
                        maxx.append(numpy.max(a[:, :, channelIdx]))
            else:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    if len(minx) < numChannels:
                        minx.append(numpy.min(a[:, :, :, channelIdx]))
                    if len(maxx) < numChannels:
                        maxx.append(numpy.max(a[:, :, :, channelIdx]))
    #print(("{} in vs {} out vs {} a-shape".format(inShape,outShape, a.shape)))
    if outDimLen<inDimLen:
        a = a.reshape(inShape)
    return minx, maxx

def normaliseFieldArray(a, numChannels, flatField=None, itype=TYPES["XRAY"], minx=None, maxx=None):
    inShape = a.shape
    inDimLen = len(inShape)
    a = numpy.squeeze(a)
    outShape = a.shape
    outDimLen = len(outShape)
    if itype == TYPES['XRAY']:
        if flatField!=None:
            a = numpy.clip(numpy.divide(a, flatField), 0.0, 1.0)
        else:
            if numChannels<=1:
                if minx is None:
                    minx = numpy.min(a)
                if maxx is None:
                    maxx = numpy.max(a)
                a = (a - minx) / (maxx - minx)
            else:
                if minx is None:
                    minx = []
                if maxx is None:
                    maxx = []
                if outDimLen<4:
                    for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                        if len(minx)<numChannels:
                            minx.append(numpy.min(a[:, :, channelIdx]))
                        if len(maxx)<numChannels:
                            maxx.append(numpy.max(a[:, :, channelIdx]))
                        if numpy.fabs(maxx[channelIdx] - minx[channelIdx]) > 0:
                            a[:, :, channelIdx] = (a[:, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])
                else:
                    for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                        if len(minx)<numChannels:
                            minx.append(numpy.min(a[:, :, :, channelIdx]))
                        if len(maxx)<numChannels:
                            maxx.append(numpy.max(a[:, :, :, channelIdx]))
                        if numpy.fabs(maxx[channelIdx] - minx[channelIdx]) > 0:
                            a[:, :, :, channelIdx] = (a[:, :, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])
    elif itype == TYPES['SCATTER']:
        if numChannels<=1:
            if minx is None:
                minx = 0
            if maxx is None:
                maxx = numpy.max(numpy.abs(a))
            a = (a-minx)/(maxx-minx)
        else:
            if minx is None:
                minx = []
            if maxx is None:
                maxx = []
            if outDimLen < 4:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    #minx.append(numpy.min(flatField[:, :, channelIdx]))
                    #maxx.append(numpy.max(flatField[:, :, channelIdx]))
                    if len(minx) < numChannels:
                        minx.append(0)
                    if len(maxx) < numChannels:
                        maxx.append(numpy.max(numpy.abs(a[:, :, channelIdx])))
                    if numpy.fabs(maxx[channelIdx] - minx[channelIdx]) > 0:
                        a[:, :, channelIdx] = (a[:, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])
            else:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    #minx.append(numpy.min(flatField[:, :, channelIdx]))
                    #maxx.append(numpy.max(flatField[:, :, channelIdx]))
                    if len(minx) < numChannels:
                        minx.append(0)
                    if len(maxx) < numChannels:
                        maxx.append(numpy.max(numpy.abs(a[:, :, :, channelIdx])))
                    if numpy.fabs(maxx[channelIdx] - minx[channelIdx]) > 0:
                        a[:, :, :, channelIdx] = (a[:, :, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])
    elif itype == TYPES['CT']:
#        if maxx:
#            if outDimLen < 4:
#                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
#                    aa=a[:, :, channelIdx]
#                    aa[aa>maxx[channelIdx]] = maxx[channelIdx]
#                    a[:, :, channelIdx] = aa
#            else:
#                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):                
#                    aa=a[:, :, :, channelIdx]
#                    aa[aa>maxx[channelIdx]] = maxx[channelIdx]
#                    a[:, :, :, channelIdx] = aa   
        if numChannels<=1:
            if minx is None:
                minx = numpy.min(a)
            if maxx is None:
                maxx = numpy.max(a)
            a = (a - minx) / (maxx-minx)
        else:
            if minx is None:
                minx = []
            if maxx is None:
                maxx = []
            if outDimLen < 4:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    if len(minx) < numChannels:
                        minx.append(numpy.min(a[:, :, channelIdx]))
                    if len(maxx) < numChannels:
                        maxx.append(numpy.max(a[:, :, channelIdx]))
                    if numpy.fabs(maxx[channelIdx] - minx[channelIdx]) > 0:
                        a[:, :, channelIdx] = (a[:, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])
                    aa = a[:, :, channelIdx]
                    aa[aa>1]=1
                    a[:, :, channelIdx]=aa
            else:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):                
                    if len(minx) < numChannels:
                        minx.append(numpy.min(a[:, :, :, channelIdx]))
                    if len(maxx) < numChannels:
                        maxx.append(numpy.max(a[:, :, :, channelIdx]))
                    if numpy.fabs(maxx[channelIdx] - minx[channelIdx]) > 0:
                        a[:, :, :, channelIdx] = (a[:, :, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])
                    aa = a[:, :, :, channelIdx]
                    aa[aa>1]=1
                    a[:, :, :, channelIdx]=aa                        
    #print(("{} in vs {} out vs {} a-shape".format(inShape,outShape, a.shape)))
    if outDimLen<inDimLen:
        a = a.reshape(inShape)
    return minx, maxx, a

def denormaliseFieldArray(a, numChannels, minx=None, maxx=None, flatField=None, itype=TYPES["XRAY"]):
    inShape = a.shape
    inDimLen = len(inShape)
    a = numpy.squeeze(a)
    outShape = a.shape
    outDimLen = len(outShape)
    if itype == TYPES['XRAY']:
        if flatField != None:
            a = a * flatField
        else:
            if numChannels <= 1:
                a = a * (maxx - minx) + minx
            else:
                if outDimLen < 4:
                    for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                        a[:, :, channelIdx] = a[:, :, channelIdx] * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
                else:
                    for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                        a[:, :, :, channelIdx] = a[:, :, :, channelIdx] * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
    elif itype == TYPES['SCATTER']:
        if numChannels <= 1:
            a = a*(maxx-minx)+minx
        else:
            if outDimLen < 4:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    a[:, :, channelIdx] = a[:, :, channelIdx] * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
            else:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    a[:, :, :, channelIdx] = a[:, :, :, channelIdx] * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
    elif itype == TYPES['CT']:
        if numChannels <= 1:
            a = a * (maxx - minx) + minx
        else:
            if outDimLen < 4:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    a[:, :, channelIdx] = a[:, :, channelIdx] * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
            else:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    a[:, :, :, channelIdx] = a[:, :, :, channelIdx] * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
    if outDimLen<inDimLen:
        a = a.reshape(inShape)
    return a



def clipNormFieldArray(a, numChannels, flatField=None, itype=TYPES["XRAY"]):
    inShape = a.shape
    inDimLen = len(inShape)
    a = numpy.squeeze(a)
    outShape = a.shape
    outDimLen = len(outShape)
    minx = None
    maxx = None

    if numChannels <= 1:
        minx = 0
        maxx = 200.0
        a = numpy.clip(a,minx,maxx)
    else:
        minx = numpy.zeros(32, a.dtype)
        # IRON-CAP
        #maxx = numpy.array([197.40414602,164.05027316,136.52565589,114.00212036,95.65716526,80.58945472,67.46342114,56.09140486,45.31409774,37.64459755,31.70887797,26.41859429,21.9482954,18.30031205,15.31461954,12.82080624,10.70525853,9.17048875,7.82142154,6.7137903,5.82180097,5.01058597,4.41808895,3.81359458,3.40606635,3.01021494,2.76689262,2.47842852,2.32304044,2.12137244,15.00464109,33.07879503], a.dtype)
        # SCANDIUM-CAP
        maxx = numpy.array([40.77704764,33.66334239,27.81001556,22.99326739,19.0324146,15.68578289,12.92738646,10.67188431,8.82938367,7.32395058,6.09176207,5.08723914,4.25534357,3.58350332,3.0290752,2.57537332,2.20811373,1.8954763,1.64371557,1.43238593,1.27925194,1.24131863,1.11358517,1.08348688,1.03346652,0.95083789,0.9814638,0.87886442,1.06108008,1.4744603,1.37953941,1.33551697])
        for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
            a[:, :, channelIdx] = numpy.clip(a[:, :, channelIdx], minx[channelIdx], maxx[channelIdx])

    if itype == TYPES['XRAY']:
        if flatField!=None:
            a = numpy.clip(numpy.divide(a, flatField), 0.0, 1.0)
        else:
            if numChannels<=1:
                a = ((a - minx) / (maxx - minx)) * 2.0 - 1.0
            else:
                if outDimLen < 4:
                    for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                        if numpy.fabs(maxx[channelIdx] - minx[channelIdx]) > 0:
                            a[:, :, channelIdx] = ((a[:, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])) * 2.0 - 1.0
                else:
                    for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                        if numpy.fabs(maxx[channelIdx] - minx[channelIdx]) > 0:
                            a[:, :, :, channelIdx] = ((a[:, :, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])) * 2.0 - 1.0
    elif itype == TYPES['SCATTER']:
        if numChannels<=1:
            a = ((a-minx)/(maxx-minx)) * 2.0 - 1.0
        else:
            if outDimLen < 4:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    if numpy.fabs(maxx[channelIdx] - minx[channelIdx]) > 0:
                        a[:, :, channelIdx] = ((a[:, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])) * 2.0 - 1.0
            else:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    if numpy.fabs(maxx[channelIdx] - minx[channelIdx]) > 0:
                        a[:, :, :, channelIdx] = ((a[:, :, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])) * 2.0 - 1.0
    elif itype == TYPES['CT']:
        if numChannels<=1:
            a = ((a - minx) / (maxx-minx)) * 2.0 - 1.0
        else:
            if outDimLen < 4:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    if numpy.fabs(maxx[channelIdx] - minx[channelIdx]) > 0:
                        a[:, :, channelIdx] = ((a[:, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])) * 2.0 - 1.0
            else:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    if numpy.fabs(maxx[channelIdx] - minx[channelIdx]) > 0:
                        a[:, :, :, channelIdx] = ((a[:, :, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])) * 2.0 - 1.0
    #print(("{} in vs {} out vs {} a-shape".format(inShape,outShape, a.shape)))
    if outDimLen<inDimLen:
        a = a.reshape(inShape)
    return minx, maxx, a

def denormFieldArray(a, numChannels, minx=None, maxx=None, flatField=None, itype=TYPES["XRAY"]):
    inShape = a.shape
    inDimLen = len(inShape)
    a = numpy.squeeze(a)
    outShape = a.shape
    outDimLen = len(outShape)
    if itype == TYPES['XRAY']:
        if flatField != None:
            a = a * flatField
        else:
            if numChannels <= 1:
                a = ((a + 1.0) / 2.0) * (maxx - minx) + minx
            else:
                if outDimLen < 4:
                    for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                        a[:, :, channelIdx] = ((a[:, :, channelIdx] + 1.0) / 2.0) * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
                else:
                    for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                        a[:, :, :, channelIdx] = ((a[:, :, :, channelIdx] + 1.0) / 2.0) * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
    elif itype == TYPES['SCATTER']:
        if numChannels <= 1:
            a = ((a + 1.0) / 2.0) * (maxx - minx) + minx
        else:
            if outDimLen < 4:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    a[:, :, channelIdx] = ((a[:, :, channelIdx] + 1.0) / 2.0) * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
            else:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    a[:, :, :, channelIdx] = ((a[:, :, :, channelIdx] + 1.0) / 2.0) * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
    elif itype == TYPES['CT']:
        if numChannels <= 1:
            a = ((a + 1.0) / 2.0) * (maxx - minx) + minx
        else:
            if outDimLen < 4:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    a[:, :, channelIdx] = ((a[:, :, channelIdx] + 1.0) / 2.0) * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
            else:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    a[:, :, :, channelIdx] = ((a[:, :, :, channelIdx] + 1.0) / 2.0) * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
    if outDimLen<inDimLen:
        a = a.reshape(inShape)
    return a


class ScatterPhantomGenerator(Sequence):
    
    def __init__(self, batch_size=1, image_size=(128, 128), input_channels=32, target_size=(128, 128), output_channels=1, useResize=False,
                 useCrop=False, useZoom=False, zoom_factor_range=(0.95,1.05), useAWGN = False, useMedian=False, useGaussian=False,
                 useFlipping=False, useNormData=False, cache=None, save_to_dir=None, save_format="png", threadLockVar=None, useCache=False):
        self.x_type=TYPES["CT"]
        self.y_type=TYPES["CT"]
        self.batch_size = batch_size
        self.image_size = image_size
        self.target_size = target_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.x_dtype_in = None
        self.y_dtype_in = None
        self.useResize = useResize
        self.useCrop = useCrop

        self.useAWGN = useAWGN
        if self.x_type==TYPES["XRAY"]:
            if self.input_channels>1:
                self.MECTnoise_mu = numpy.array(
                    [0.26358, 0.24855, 0.23309, 0.22195, 0.21639, 0.21285, 0.21417, 0.21979, 0.22502, 0.23387, 0.24120,
                     0.24882, 0.25177, 0.25594, 0.26005, 0.26350, 0.27067, 0.27440, 0.27284, 0.26868, 0.26477, 0.25461,
                     0.24436, 0.24287, 0.23849, 0.24022, 0.23915, 0.23874, 0.23968, 0.23972, 0.24100, 0.23973, 0.23921,
                     0.24106, 0.24177, 0.24155, 0.24358, 0.24578, 0.24682, 0.24856, 0.24969, 0.25206, 0.25337, 0.25650,
                     0.25627, 0.25921, 0.26303, 0.26615, 0.26772, 0.26882, 0.27248, 0.27400, 0.27722, 0.27905, 0.28138,
                     0.28406, 0.28593, 0.28830, 0.29129, 0.29420, 0.29673, 0.29776, 0.29955, 0.30050, 0.30151, 0.30196,
                     0.30340, 0.30282, 0.30546, 0.30509, 0.30569, 0.30667, 0.30512, 0.30413, 0.30496, 0.30474, 0.30525,
                     0.30534, 0.30503, 0.30635, 0.30539, 0.30561, 0.30660, 0.30491, 0.30486, 0.30291, 0.30323, 0.30253,
                     0.29960, 0.29734, 0.29760, 0.29464, 0.29273, 0.29035, 0.28906, 0.28680, 0.28446, 0.27905, 0.27842,
                     0.27555, 0.27112, 0.26879, 0.26760, 0.26547, 0.26289, 0.25914, 0.25776, 0.25641, 0.25394, 0.25148,
                     0.25033, 0.24752, 0.24648, 0.24424, 0.24386, 0.24097, 0.24095, 0.24104, 0.24090, 0.23948, 0.23985,
                     0.23916, 0.23931, 0.23869, 0.23922, 0.23671, 0.23994, 0.24009, 0.24299, 0.25392, 0.26096, 0.26740,
                     0.27136, 0.27207, 0.27209, 0.26671, 0.26037, 0.25427, 0.25223, 0.25006, 0.24506, 0.23531, 0.22816,
                     0.21955, 0.21713, 0.21705, 0.22167, 0.23419, 0.24789, 0.26416], dtype=numpy.float32)
                self.MECTnoise_sigma = numpy.array(
                    [0.03491, 0.02537, 0.01526, 0.00798, 0.00368, 0.00220, 0.00389, 0.00819, 0.01553, 0.02466, 0.03281,
                     0.03765, 0.04221, 0.04212, 0.03958, 0.03447, 0.02916, 0.02766, 0.02757, 0.02671, 0.02047, 0.01121,
                     0.00309, 0.00321, 0.00397, 0.00433, 0.00456, 0.00514, 0.00598, 0.00674, 0.00784, 0.00852, 0.00934,
                     0.01040, 0.01126, 0.01257, 0.01349, 0.01460, 0.01611, 0.01770, 0.01930, 0.02084, 0.02267, 0.02354,
                     0.02597, 0.02677, 0.02758, 0.02859, 0.03009, 0.03126, 0.03121, 0.03174, 0.03198, 0.03140, 0.03225,
                     0.03150, 0.03105, 0.03154, 0.03063, 0.02965, 0.02933, 0.02837, 0.02748, 0.02662, 0.02632, 0.02540,
                     0.02497, 0.02515, 0.02463, 0.02431, 0.02462, 0.02559, 0.02677, 0.02757, 0.02812, 0.02728, 0.02712,
                     0.02635, 0.02568, 0.02622, 0.02636, 0.02611, 0.02635, 0.02649, 0.02604, 0.02533, 0.02588, 0.02643,
                     0.02724, 0.02824, 0.02925, 0.02916, 0.02922, 0.03064, 0.03059, 0.03050, 0.03066, 0.03251, 0.03196,
                     0.03219, 0.03295, 0.03199, 0.03130, 0.02980, 0.02977, 0.02886, 0.02701, 0.02579, 0.02406, 0.02252,
                     0.02103, 0.01931, 0.01750, 0.01566, 0.01390, 0.01238, 0.01035, 0.00918, 0.00798, 0.00687, 0.00606,
                     0.00523, 0.00467, 0.00423, 0.00397, 0.00430, 0.00411, 0.00344, 0.00222, 0.00929, 0.01874, 0.02600,
                     0.02750, 0.02828, 0.02741, 0.03276, 0.03759, 0.04272, 0.04187, 0.03968, 0.03494, 0.02768, 0.01931,
                     0.01083, 0.00532, 0.00377, 0.00777, 0.01520, 0.02568, 0.03590], dtype=numpy.float32)
            else:
                self.SECTnoise_mu = 0.2638478
                self.SECTnoise_sigma = 0.022102864
        elif self.x_type==TYPES["SCATTER"]:
            if self.input_channels>1:
                self.MECTnoise_mu = numpy.array(
                    [0.16866670, 0.17188519, 0.17515915, 0.17835625, 0.18134183, 0.18390702, 0.18609122, 0.18790514,
                     0.18933285, 0.19041915, 0.19118198, 0.19161633, 0.19181411, 0.19186348, 0.19185040, 0.19184256,
                     0.19188470, 0.19197313, 0.19209580, 0.19224754, 0.19241207, 0.19258621, 0.19275952, 0.19290241,
                     0.19301524, 0.19307717, 0.19306810, 0.19298917, 0.19281435, 0.19255346, 0.19221039, 0.19176093,
                     0.19122621, 0.19061593, 0.18991660, 0.18913179, 0.18829705, 0.18740011, 0.18644242, 0.18542260,
                     0.18435654, 0.18325110, 0.18211837, 0.18094630, 0.17975648, 0.17856240, 0.17736573, 0.17617767,
                     0.17500219, 0.17385002, 0.17272437, 0.17162455, 0.17057456, 0.16956633, 0.16860679, 0.16770124,
                     0.16684261, 0.16603374, 0.16528716, 0.16458883, 0.16394797, 0.16335864, 0.16282625, 0.16234653,
                     0.16191163, 0.16152307, 0.16117932, 0.16088236, 0.16062331, 0.16040275, 0.16022629, 0.16007563,
                     0.15996251, 0.15987573, 0.15982041, 0.15979193, 0.15978788, 0.15981399, 0.15986311, 0.15994251,
                     0.16005459, 0.16019302, 0.16036940, 0.16058268, 0.16083441, 0.16112886, 0.16147478, 0.16187346,
                     0.16231956, 0.16282015, 0.16338502, 0.16399501, 0.16466099, 0.16538820, 0.16617289, 0.16701784,
                     0.16791800, 0.16887256, 0.16987358, 0.17093329, 0.17204428, 0.17319362, 0.17438019, 0.17559014,
                     0.17682701, 0.17806635, 0.17931469, 0.18056600, 0.18180696, 0.18301938, 0.18419762, 0.18533589,
                     0.18641122, 0.18743520, 0.18839959, 0.18929145, 0.19010590, 0.19084049, 0.19149243, 0.19206308,
                     0.19254355, 0.19291977, 0.19319933, 0.19338428, 0.19347334, 0.19348168, 0.19342172, 0.19330871,
                     0.19316624, 0.19299663, 0.19281704, 0.19264048, 0.19248135, 0.19235786, 0.19225717, 0.19219786,
                     0.19214044, 0.19203097, 0.19180060, 0.19132749, 0.19051121, 0.18933457, 0.18781172, 0.18595189,
                     0.18373764, 0.18111646, 0.17815353, 0.17494756, 0.17167550, 0.16848442], dtype=numpy.float32)
                self.MECTnoise_sigma = numpy.array(
                    [0.03945668, 0.03752889, 0.03544898, 0.03323879, 0.03096720, 0.02856092, 0.02612186, 0.02376646,
                     0.02169589, 0.02004326, 0.01889315, 0.01825106, 0.01797775, 0.01793787, 0.01798946, 0.01807670,
                     0.01817564, 0.01827498, 0.01837080, 0.01845804, 0.01854509, 0.01863035, 0.01872147, 0.01882850,
                     0.01894833, 0.01909700, 0.01926717, 0.01944903, 0.01965817, 0.01988674, 0.02013096, 0.02037972,
                     0.02064439, 0.02092013, 0.02119017, 0.02143954, 0.02169283, 0.02193263, 0.02214348, 0.02231043,
                     0.02243537, 0.02251225, 0.02254155, 0.02250185, 0.02240452, 0.02225120, 0.02204627, 0.02180226,
                     0.02151376, 0.02118313, 0.02082264, 0.02043567, 0.02004489, 0.01964261, 0.01923366, 0.01882669,
                     0.01841781, 0.01800790, 0.01761821, 0.01723964, 0.01688061, 0.01654472, 0.01623485, 0.01595351,
                     0.01568520, 0.01544452, 0.01522702, 0.01503263, 0.01486395, 0.01472051, 0.01461037, 0.01451426,
                     0.01445295, 0.01441048, 0.01440137, 0.01441387, 0.01444312, 0.01449269, 0.01456381, 0.01465286,
                     0.01476782, 0.01489762, 0.01505751, 0.01523783, 0.01543211, 0.01564785, 0.01588430, 0.01613858,
                     0.01641310, 0.01670377, 0.01701734, 0.01733008, 0.01765341, 0.01798678, 0.01833384, 0.01867848,
                     0.01902556, 0.01936452, 0.01968341, 0.02000430, 0.02031607, 0.02061045, 0.02088021, 0.02110307,
                     0.02129998, 0.02145134, 0.02156904, 0.02166331, 0.02172182, 0.02173031, 0.02169494, 0.02162645,
                     0.02150349, 0.02133820, 0.02113991, 0.02091581, 0.02066636, 0.02039382, 0.02011436, 0.01984492,
                     0.01957840, 0.01930333, 0.01905621, 0.01882268, 0.01862423, 0.01844558, 0.01829097, 0.01816721,
                     0.01807584, 0.01799463, 0.01795434, 0.01791797, 0.01787606, 0.01783821, 0.01779470, 0.01775097,
                     0.01772077, 0.01774389, 0.01789813, 0.01839385, 0.01945910, 0.02104492, 0.02306657, 0.02539757,
                     0.02782046, 0.03022519, 0.03256747, 0.03480967, 0.03695302, 0.03895815], dtype=numpy.float32)
            else:
                self.SECTnoise_mu = 0.17900705
                self.SECTnoise_sigma = 0.020096453

        self.useZoom = useZoom
        self.zoom_factor_range = zoom_factor_range
        self.useFlipping = useFlipping
        self.useMedian = useMedian
        self.medianSize = [0,1,3,5,7,9,11]
        self.useGaussian = useGaussian
        self.gaussianRange = (0, 0.075)
        self.useNormData = useNormData
        self._epoch_num_ = 0
        self.numImages = 0

        #self.numImages = 0
        #dims = (target_size[0], target_size[1], input_channels)
        #self.targetSize = (self.targetSize[0], self.targetSize[1], self.image_size[2])
        # ========================================#
        # == zoom-related image information ==#
        # ========================================#
        self.im_center = None
        self.im_shift = None
        self.im_bounds = None
        self.im_center = numpy.array([int(self.target_size[0] - 1) / 2, int(self.target_size[1] - 1) / 2], dtype=numpy.int32)
        self.im_shift = numpy.array([(self.image_size[0] - 1) / 2, (self.image_size[1] - 1) / 2], dtype=numpy.int32)
        left = max(self.im_shift[0] - self.im_center[0],0)
        right = min(left + self.target_size[0],self.image_size[0])
        top = max(self.im_shift[1] - self.im_center[1],0)
        bottom = min(top + self.target_size[1],self.image_size[1])
        self.im_bounds = (left, right, top, bottom)

        #===================================#
        #== directory-related information ==#
        #===================================#
        self.fileArray = []
        self.in_dir = ""
        self.out_dir = ""
        self.save_to_dir=save_to_dir
        self.save_format=save_format
        #==================================#
        #== flat-field related variables ==#
        #==================================#
        self.flatField_input = None
        self.flatField_output = None
        #===============================#
        #== caching-related variables ==#
        #===============================#
        self.useCache = useCache
        self.cache = cache
        self._lock_ = threadLockVar
        self.seeded = False
        #======================#
        #== batch size setup ==#
        #======================#
        self.batch_image_size_X = (self.batch_size, self.image_size[0], self.image_size[1], self.input_channels, 1)
        self.batch_image_size_Y = (self.batch_size, self.image_size[0], self.image_size[1], self.output_channels, 1)
        if self.useCrop or self.useResize:
            self.batch_image_size_X = (self.batch_size, self.target_size[0], self.target_size[1], self.input_channels, 1)
            self.batch_image_size_Y = (self.batch_size, self.target_size[0], self.target_size[1], self.output_channels, 1)
        
    def prepareDirectFileInput(self, input_image_paths, flatFieldFilePath=None):
        for entry in input_image_paths:
            for name in glob.glob(os.path.join(entry, '*.h5')):
                self.fileArray.append(name)
        digits = re.compile(r'(\d+)')
        def tokenize(filename):
            return tuple(int(token) if match else token for token, match in
                         ((fragment, digits.search(fragment)) for fragment in digits.split(filename)))
        # = Now you can sort your file names like so: =#
        self.fileArray.sort(key=tokenize)
        self.numImages = len(self.fileArray)

        # === prepare image sizes === #
        inImgDims = (self.image_size[0], self.image_size[1], self.input_channels, 1)
        outImgDims = (self.image_size[0], self.image_size[1], self.output_channels, 1)

        if len(self.fileArray) and os.path.exists(self.fileArray[0]):
            self.in_dir = os.path.dirname(self.fileArray[0])
            f = h5py.File(self.fileArray[0], 'r')
            # define your variable names in here
            imX = numpy.array(f['Data_X'], order='F').transpose()
            f.close()
            self.x_dtype_in = imX.dtype
            if len(imX.shape) > 3:
                imX = numpy.squeeze(imX[:, :, 0, :])
            if len(imX.shape) < 3:
                imX = imX.reshape(imX.shape + (1,))
            # === we need to feed the data as 3D+1 channel data stack === #
            if len(imX.shape) < 4:
                imX = imX.reshape(imX.shape + (1,))

            if imX.shape != inImgDims:
                print("Error - read data shape ({}) and expected data shape ({}) of X are not equal. EXITING ...".format(imX.shape, inImgDims))
                exit()

        if len(self.fileArray) and os.path.exists(self.fileArray[0]):
            f = h5py.File(self.fileArray[0], 'r')
            # define your variable names in here
            imY = numpy.array(f['Data_Y'], order='F').transpose()
            f.close()
            self.y_dtype_in = imY.dtype
            if len(imY.shape) > 3:
                imY = numpy.squeeze(imY[:,:,0,:])
            if len(imY.shape) < 3:
                imY = imY.reshape(imY.shape + (1,))
            # === we need to feed the data as 3D+1 channel data stack === #
            if len(imY.shape) < 4:
                imY = imY.reshape(imY.shape + (1,))

            if imY.shape != outImgDims:
                print("Error - read data shape ({}) and expected data shape ({}) of X are not equal. EXITING ...".format(imX.shape,inImgDims))
                exit()

        # ======================================== #
        # ==== crop-related image information ==== #
        # ======================================== #
        self.im_center = None
        self.im_shift = None
        self.im_bounds = None
        self.im_center = numpy.array([int(self.target_size[0] - 1) / 2, int(self.target_size[1] - 1) / 2], dtype=numpy.int32)
        self.im_shift = numpy.array([(self.image_size[0] - 1) / 2, (self.image_size[1] - 1) / 2], dtype=numpy.int32)
        left = max(self.im_shift[0] - self.im_center[0],0)
        right = min(left + self.target_size[0],self.image_size[0])
        top = max(self.im_shift[1] - self.im_center[1],0)
        bottom = min(top + self.target_size[1],self.image_size[1])
        self.im_bounds = (left, right, top, bottom)

        # === prepare flat-field normalization === #
        if (flatFieldFilePath != None) and (len(flatFieldFilePath) > 0):
            f = h5py.File(flatFieldFilePath, 'r')
            self.flatField_output = numpy.array(f['data']['value'])  # f['data0']
            f.close()
            self.flatField_input = numpy.array(self.flatField_output)
            if self.output_channels==1:
                #self.flatField_output = numpy.sum(self.flatField_output,2)
                self.flatField_input = numpy.mean(self.flatField_output, 2)

    def _initCache_locked_(self):
        startId = 0
        loadData_flag = True
        while (loadData_flag):
            ii = 0
            with self._lock_:
                loadData_flag = (self.cache.is_cache_updated() == False)
                ii=self.cache.get_renew_index()
            if loadData_flag == False:
                break
            file_index = numpy.random.randint(0, self.numImages)
            inName = self.fileArray[file_index]
            f = h5py.File(inName, 'r')
            imX = numpy.array(f['Data_X'], order='F').transpose()
            imY = numpy.array(f['Data_Y'], order='F').transpose()
            f.close()
            if len(imX.shape) != len(imY.shape):
                print("Image dimensions do not match - EXITING ...")
                exit(1)

            if len(imX.shape) > 3:
                indices = [ii,]
                while (len(indices)<imX.shape[2]) and (loadData_flag==True):
                    with self._lock_:
                        ii = self.cache.get_renew_index()
                        indices.append(ii)
                    with self._lock_:
                        loadData_flag = (self.cache.is_cache_updated() == False)
                slice_indices = numpy.random.randint(0, imX.shape[2],len(indices))
                """
                Data Normalisation
                """
                minValX = None
                maxValX = None
                minValY = None
                maxValY = None
                if self.useNormData:
                    """
                    Data Normalisation
                    """
                    #Norm by X only
#                    minValX, maxValX, imX = normaliseFieldArray(imX, self.input_channels, self.flatField_input, self.x_type)
#                    minValY, maxValY, imY = normaliseFieldArray(imY, self.output_channels, self.flatField_output, self.y_type, minx=minValX, maxx=maxValX)
                    #Norm by Ti attenuation
                    mx = ([0, 0, 0, 0, 0, 0, 0, 0])
                    mu_Ti = ([23.7942, 6.7035, 3.1118, 1.8092, 1.2668, 0.9963, 0.8428, 0.7473])
                    minValX, maxValX, imX = normaliseFieldArray(imX, self.input_channels, self.flatField_input, self.x_type, minx=mx, maxx=mu_Ti)
                    minValY, maxValY, imY = normaliseFieldArray(imY, self.output_channels, self.flatField_output, self.y_type, minx=mx, maxx=mu_Ti)

                for index in itertools.islice(itertools.count(), 0, len(indices)):
                    slice_index = slice_indices[index]
                    imX_slice = numpy.squeeze(imX[:,:,slice_index,:])
                    # === we need to feed the data as 3D+1 channel data stack === #
                    if len(imX_slice.shape) < 3:
                        imX_slice = imX_slice.reshape(imX_slice.shape + (1,))
                    if len(imX_slice.shape) < 4:
                        imX_slice = imX_slice.reshape(imX_slice.shape + (1,))
                    imY_slice = numpy.squeeze(imY[:,:,slice_index,:])
                    if len(imY_slice.shape) < 3:
                        imY_slice = imY_slice.reshape(imY_slice.shape + (1,))
                    if len(imY_slice.shape) < 4:
                        imY_slice = imY_slice.reshape(imY_slice.shape + (1,))
                    # == Note: do data normalization here to reduce memory footprint ==#
                    #"""
                    #Data Normalisation
                    #"""
                    #if self.useNormData:
                    #    minValX, maxValX, imX_slice = normaliseFieldArray(imX_slice, self.input_channels, self.flatField_input, self.x_type)
                    #    imX_slice = imX_slice.astype(numpy.float32)
                    imX_slice = imX_slice.astype(numpy.float32)
                    imY_slice = imY_slice.astype(numpy.float32)
                    with self._lock_:
                        self.cache.set_cache_item_x(indices[index],imX_slice)
                        self.cache.set_item_limits_x(indices[index], minValX, maxValX)
                        self.cache.set_cache_item_y(indices[index],imY_slice)
                        self.cache.set_item_limits_y(indices[index], minValY, maxValY)

            else:
                # === we need to feed the data as 3D+1 channel data stack === #
                if len(imX.shape) < 3:
                    imX = imX.reshape(imX.shape + (1,))
                if len(imX.shape) < 4:
                    imX = imX.reshape(imX.shape + (1,))
                if len(imY.shape) < 3:
                    imY = imY.reshape(imY.shape + (1,))
                if len(imY.shape) < 4:
                    imY = imY.reshape(imY.shape + (1,))
                # == Note: do data normalization here to reduce memory footprint ==#
                """
                Data Normalisation
                """
                minValX = None
                maxValX = None
                minValY = None
                maxValY = None
                if self.useNormData:
                    #Norm by X only
#                    minValX, maxValX, imX = normaliseFieldArray(imX, self.input_channels, self.flatField_input, self.x_type)
#                    minValY, maxValY, imY = normaliseFieldArray(imY, self.output_channels, self.flatField_output, self.y_type, minx=minValX, maxx=maxValX)
                    #Norm by Ti attenuation
                    mx = ([0, 0, 0, 0, 0, 0, 0, 0])
                    mu_Ti = ([23.7942, 6.7035, 3.1118, 1.8092, 1.2668, 0.9963, 0.8428, 0.7473])
                    minValX, maxValX, imX = normaliseFieldArray(imX, self.input_channels, self.flatField_input, self.x_type, minx=mx, maxx=mu_Ti)
                    minValY, maxValY, imY = normaliseFieldArray(imY, self.output_channels, self.flatField_output, self.y_type, minx=mx, maxx=mu_Ti)
                imX = imX.astype(numpy.float32)
                imY = imY.astype(numpy.float32)
                with self._lock_:
                    self.cache.set_cache_item_x(ii,imX)
                    self.cache.set_item_limits_x(ii, minValX, maxValX)
                    self.cache.set_cache_item_y(ii, imY)
                    self.cache.set_item_limits_y(ii, minValY, maxValY)

            with self._lock_:
                loadData_flag = (self.cache.is_cache_updated() == False)
        return



    def __len__(self):
        epoch_steps = 4096
        #batchSize = 8
        return int(epoch_steps/self.batch_size)
    
    def __getitem__(self, idx):
        pid = os.getpid()
        if self.seeded == False:
            numpy.random.seed(pid)
            random.seed(pid)
            self.seeded=True

        if self.useCache:
            flushCache = False
            with self._lock_:
                flushCache = (self.cache.is_cache_updated() == False)
            if flushCache == True:
                self._initCache_locked_()

        batchX = numpy.zeros(self.batch_image_size_X, dtype=numpy.float32)
        batchY = numpy.zeros(self.batch_image_size_Y, dtype=numpy.float32)
        idxArray = numpy.random.randint(0, self.cache.get_cache_size(), self.batch_size)
        for j in itertools.islice(itertools.count(),0,self.batch_size):
            imX = None
            minValX = None
            maxValX = None
            imY = None
            minValY = None
            maxValY = None
            if self.useCache:
                #imgIndex = numpy.random.randint(0, self.cache_size)
                imgIndex =idxArray[j]
                with self._lock_:
                    imX = self.cache.get_cache_item_x(imgIndex)
                    minValX, maxValX = self.cache.get_item_limits_x(imgIndex)
                    imY = self.cache.get_cache_item_y(imgIndex)
                    minValY, maxValY = self.cache.get_item_limits_y(imgIndex)
            else:
                # imgIndex = min([(idx*self.batch_size)+j, self.numImages-1,len(self.inputFileArray)-1])
                imgIndex = ((idx * self.batch_size) + j) % (self.numImages - 1)
                """
                Load data from disk
                """
                inName = self.fileArray[imgIndex]
                f = h5py.File(inName, 'r')
                imX = numpy.array(f['Data_X'], order='F').transpose()
                imY = numpy.array(f['Data_Y'], order='F').transpose()
                f.close()
                if len(imX.shape) < 3:
                    imX = imX.reshape(imX.shape + (1,))
                if len(imX.shape) < 4:
                    imX = imX.reshape(imX.shape + (1,))
                if len(imY.shape) < 3:
                    imY = imY.reshape(imY.shape + (1,))
                if len(imY.shape) < 4:
                    imY = imY.reshape(imY.shape + (1,))

                if imX.shape != imY.shape:
                    raise RuntimeError("Input- and Output sizes do not match.")
                # == Note: do data normalization here to reduce memory footprint ==#
                """
                Data Normalisation
                """
                if self.useNormData:
                    #Norm by X only
#                    minValX, maxValX, imX = normaliseFieldArray(imX, self.input_channels, self.flatField_input, self.x_type)
#                    minValY, maxValY, imY = normaliseFieldArray(imY, self.output_channels, self.flatField_output, self.y_type, minx=minValX, maxx=maxValX)
                    #Norm by Ti attenuation
                    mx = ([0, 0, 0, 0, 0, 0, 0, 0])
                    mu_Ti = ([23.7942, 6.7035, 3.1118, 1.8092, 1.2668, 0.9963, 0.8428, 0.7473])
                    minValX, maxValX, imX = normaliseFieldArray(imX, self.input_channels, self.flatField_input, self.x_type, minx=mx, maxx=mu_Ti)
                    minValY, maxValY, imY = normaliseFieldArray(imY, self.output_channels, self.flatField_output, self.y_type, minx=mx, maxx=mu_Ti)

                imX = imX.astype(numpy.float32)
                imY = imY.astype(numpy.float32)

            fname_in = "img_{}_{}_{}".format(self._epoch_num_, idx, j)
            """
            Data augmentation
            """
            input_target_size = self.target_size + (self.input_channels, )
            output_target_size = self.target_size + (self.output_channels, )
            if self.useZoom:
                _zoom_factor = numpy.random.uniform(self.zoom_factor_range[0], self.zoom_factor_range[1])
                for channelIdx in itertools.islice(itertools.count(), 0, self.input_channels):
                    imX[:,:,channelIdx] = clipped_zoom(imX[:,:,channelIdx], _zoom_factor, order=3, mode='reflect')
                    imY[:,:,channelIdx] = clipped_zoom(imY[:,:,channelIdx], _zoom_factor, order=3, mode='reflect')
            if self.useResize:
                imX = transform.resize(imX, input_target_size, order=3, mode='reflect')
                imY = transform.resize(imY, output_target_size, order=3, mode='constant')
            if self.useCrop:
                imX = imX[self.im_bounds[0]:self.im_bounds[1], self.im_bounds[2]:self.im_bounds[3],:]
                imY = imY[self.im_bounds[0]:self.im_bounds[1], self.im_bounds[2]:self.im_bounds[3],:]
            if self.useFlipping:
                mode = numpy.random.randint(0,4)
                if mode == 0:   # no modification
                    pass
                if mode == 1:
                    imX = numpy.fliplr(imX)
                    imY = numpy.fliplr(imY)
                if mode == 2:
                    imX = numpy.flipud(imX)
                    imY = numpy.flipud(imY)
                if mode == 3:
                    imX = numpy.fliplr(imX)
                    imX = numpy.flipud(imX)
                    imY = numpy.fliplr(imY)
                    imY = numpy.flipud(imY)
            if self.useAWGN: # only applies to input data
                if self.input_channels > 1:
                    for channelIdx in itertools.islice(itertools.count(), 0, self.input_channels):
                        rectMin = numpy.min(imX[:,:,channelIdx])
                        rectMax = numpy.max(imX[:,:,channelIdx])
                        imX[:,:,channelIdx] = util.random_noise(imX[:,:,channelIdx], mode='gaussian', mean=self.MECTnoise_mu[channelIdx]*0.15, var=(self.MECTnoise_sigma[channelIdx]*0.15*self.MECTnoise_sigma[channelIdx]*0.15))
                        imX[:,:,channelIdx] = numpy.clip(imX[:,:,channelIdx], rectMin, rectMax)
                else:
                    rectMin = numpy.min(imX[:, :, 0])
                    rectMax = numpy.max(imX[:, :, 0])
                    imX[:, :, channelIdx] = util.random_noise(imX[:, :, 0], mode='gaussian', mean=self.SECTnoise_mu * 0.15, var=(self.SECTnoise_sigma * 0.15 * self.SECTnoise_sigma * 0.15))
                    imX[:, :, channelIdx] = numpy.clip(imX[:, :, channelIdx], rectMin, rectMax)
            if self.useMedian:
                mSize = self.medianSize[numpy.random.randint(0,len(self.medianSize))]
                if mSize > 0:
                    imX = ndimage.median_filter(imX, (mSize, mSize, 1, 1), mode='constant', cval=1.0)
                    # should the output perhaps always be median-filtered ?
                    #outImgY = ndimage.median_filter(outImgY, (mSize, mSize, 1), mode='constant', cval=1.0)
            if self.y_type==TYPES["XRAY"]:
                #imY = ndimage.median_filter(imY, (3, 3, 1), mode='constant', cval=1.0)
                imY = ndimage.median_filter(imY, (3, 3, 1, 1), mode='reflect', cval=1.0)
            if self.useGaussian:
                # here, it's perhaps incorrect to also smoothen the output;
                # rationale: even an overly smooth image should result is sharp outputs
                sigma = numpy.random.uniform(low=self.gaussianRange[0], high=self.gaussianRange[1])
                imX = ndimage.gaussian_filter(imX, (sigma, sigma, 0))
                #outImgY = ndimage.gaussian_filter(outImgY, (sigma, sigma, 0))
            """
            Store data if requested
            """
            if self.save_to_dir != None:
                sXImg = array_to_img(imX[:,:,0,0])
                #save_img(os.path.join(self.save_to_dir,fname_in+"."+self.save_format),sXimg)
                fname_in = "img_"+str(imgIndex)
                sXImg.save(os.path.join(self.save_to_dir,fname_in+"."+self.save_format))
                sYImg = array_to_img(imY[:,:,0,0])
                #save_img(os.path.join(self.save_to_dir,fname_out+"."+self.save_format), sYImg)
                fname_out = "img_" + str(imgIndex)
                sYImg.save(os.path.join(self.save_to_dir,fname_out+"."+self.save_format))
            batchX[j] = imX
            batchY[j] = imY
        return batchX, batchY
    
    def on_epoch_end(self):
        self._epoch_num_ = self._epoch_num_ + 1
        # print("Epoch: {}, num. CT gens called: {}".format(self._epoch_num_, self.fan_beam_CT.getNumberOfTransformedData()))