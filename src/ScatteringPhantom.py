import numpy
#import threading
import multiprocessing
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
dims = (256,256,1)
#NORM_DATA_MODE = 0  # 0 - per image over all channels; 1 - per image per channel; 2 - flat-field norm
TYPES = {"XRAY": 0, "SCATTER": 1, "CT": 2, "prenormalized": 3}

def numpy_normalize(v):
    norm = numpy.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm

def normaliseFieldArray(a, numChannels, flatField=None, itype=TYPES["CT"]):
    minx = None
    maxx = None
    if itype == TYPES['XRAY']:
        if flatField!=None:
            a = numpy.clip(numpy.divide(a, flatField), 0.0, 1.0)
        else:
            if numChannels<=1:
                minx = numpy.min(a)
                maxx = numpy.max(a)
                a = (a - minx) / (maxx - minx)
            else:
                minx = []
                maxx = []
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    minx.append(numpy.min(a[:, :, channelIdx]))
                    maxx.append(numpy.max(a[:, :, channelIdx]))
                    a[:, :, channelIdx] = (a[:, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])
    elif itype == TYPES['SCATTER']:
        if numChannels<=1:
            minx = 0
            maxx = numpy.max(numpy.abs(a))
            a = (a-minx)/(maxx-minx)
        else:
            minx = []
            maxx = []
            for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                #minx.append(numpy.min(flatField[:, :, channelIdx]))
                #maxx.append(numpy.max(flatField[:, :, channelIdx]))
                minx.append(0)
                maxx.append(numpy.max(numpy.abs(a[:, :, channelIdx])))
                a[:, :, channelIdx] = (a[:, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])
    elif itype == TYPES['CT']:
        #NORMALIZE BY MAX MIN RANGE
        if numChannels<=1:
            minx = numpy.min(a)
            maxx = numpy.max(a)
            a = (a - minx) / (maxx-minx)
        else:
            minx = []
            maxx = []
            for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                minx.append(numpy.min(a[:, :, channelIdx]))
                maxx.append(numpy.max(a[:, :, channelIdx]))
                a[:, :, channelIdx] = (a[:, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])
#        a=numpy.exp(-a)
#        a=1-numpy.exp(-a)
    elif itype == TYPES['prenormalized']:                
        a=a                
    return minx, maxx, a

def denormaliseFieldArray(a, numChannels, minx=None, maxx=None, flatField=None, itype=TYPES["CT"]):
    if itype == TYPES['XRAY']:
        if flatField != None:
            a = a * flatField
        else:
            if numChannels <= 1:
                a = a * (maxx - minx) + minx
            else:
                for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                    a[:, :, channelIdx] = a[:, :, channelIdx] * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
    elif itype == TYPES['SCATTER']:
        if numChannels <= 1:
            a = a*(maxx-minx)+minx
        else:
            for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                a[:, :, channelIdx] = a[:, :, channelIdx] * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
    elif itype == TYPES['CT']:
#        #NORMALIZE BY MAX MIN RANGE        
        if numChannels <= 1:
            a = a * (maxx - minx) + minx
        else:
            for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                a[:, :, channelIdx] = a[:, :, channelIdx] * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
#        a=-numpy.log(1-a)            
    elif itype == TYPES['prenormalized']:                
        a=a                
    return a

class ScatterPhantomGenerator(Sequence):
    
    def __init__(self, batch_size=1, image_size=(128, 128), input_channels=32, target_size=(128, 128), output_channels=1, useResize=False,
                 useCrop=False, useZoom=False, zoomFactor=1.0, useAWGN = False, useMedian=False, useGaussian=False,
                 useFlipping=False, useNormData=False, cache_period=512, save_to_dir=None, save_format="png", threadLockVar=None, useCache=False):
        self.x_type=TYPES["CT"]
        self.y_type=TYPES["CT"]
        self.batch_size = batch_size
        self.image_size = image_size
        self.target_size = target_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dtype = numpy.float32
        self.useResize = useResize
        self.useNormData = useNormData
        self.numImages = 0
        dims = (target_size[0], target_size[1], input_channels)
        
        # ============================ #
        # Data Augmentation parameters #
        # ============================ #
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
        elif self.x_type==TYPES["CT"]:
            if self.input_channels>1:
                self.MECTnoise_mu = numpy.array(
                    [], dtype=numpy.float32)
                self.MECTnoise_sigma = numpy.array(
                    [], dtype=numpy.float32)
            else:
                self.SECTnoise_mu = 0.5
                self.SECTnoise_sigma = 0.25
        self.useCrop = useCrop
        self.useZoom = useZoom
        self.zoomFactor = zoomFactor
        self.useFlipping = useFlipping
        self.useMedian = useMedian
        self.medianSize = [0,1,3,5,7,9,11]
        self.useGaussian = useGaussian
        self.gaussianRange = (0, 0.075)
        # =================================== #
        # End of data Augmentation parameters #
        # =================================== #

        # ========================================#
        # == zoom-related image information ==#
        # ========================================#
        self.im_center = numpy.array(
            [int(self.image_size[0] * self.zoomFactor - 1) / 2, int(self.image_size[1] * self.zoomFactor - 1) / 2],
            dtype=numpy.int32)
        self.im_shift = numpy.array([(self.image_size[0] - 1) / 2, (self.image_size[1] - 1) / 2], dtype=numpy.int32)
        left = self.im_center[0] - self.im_shift[0]
        right = left + self.image_size[0]
        top = self.im_center[1] - self.im_shift[1]
        bottom = top + self.image_size[1]
        self.im_bounds = (left, right, top, bottom)
        #===================================#
        #== directory-related information ==#
        #===================================#
        self.fileArray = []
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
        self.cache_size = CACHE_SIZE
        self.cache_period = cache_period
        self.cacheX = numpy.zeros(1,dtype=numpy.float32)
        self.cacheY = numpy.zeros(1,dtype=numpy.float32)
        self.renew_cache = multiprocessing.Value(ctypes.c_bool,False)
        self.cacheUsed_counter = multiprocessing.Value('i',0)
        self.cacheRenewed_counter = multiprocessing.Value('i',0)
        self._lock_ = threadLockVar
        self._memlock_ = multiprocessing.Lock()
        self._refreshEvent_ = multiprocessing.Event()
        #======================#
        #== batch size setup ==#
        #======================#
        self.batch_image_size_X = (self.batch_size, self.image_size[0], self.image_size[1], 1)
        self.batch_image_size_Y = (self.batch_size, self.image_size[0], self.image_size[1], 1)
        if self.useCrop or self.useResize:
            self.batch_image_size_X = (self.batch_size, self.target_size[0], self.target_size[1], 1)
            self.batch_image_size_Y = (self.batch_size, self.target_size[0], self.target_size[1], 1)
        
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
        inImgDims = (self.image_size[0], self.image_size[1], self.input_channels)
        outImgDims = (self.image_size[0], self.image_size[1], self.output_channels)
        f = h5py.File(self.fileArray[0], 'r')
        # define your variable names in here
        imX = numpy.array(f['Data_X'], order='F').transpose()
        f.close()
        if len(imX.shape) < 3:
            imX = imX.reshape(imX.shape + (1,))
        if imX.shape != inImgDims:
            print("Error - read data shape and expected data shape of X are not equal. EXITING ...")
            exit()

        # === prepare caching === #
        if self.useCache:
            self.cacheX = numpy.zeros((self.cache_size, inImgDims[0], inImgDims[1], inImgDims[2]), dtype=numpy.float32)
            self.cacheY = numpy.zeros((self.cache_size, inImgDims[0], inImgDims[1], inImgDims[2]), dtype=numpy.float32)
            self.renew_cache = multiprocessing.Value(ctypes.c_bool, False)
            self.cacheUsed_counter = multiprocessing.Value('i', 0)
            self.__initCache_open_()
        # === prepare flat-field normalization === #
        if (flatFieldFilePath != None) and (len(flatFieldFilePath) > 0):
            f = h5py.File(flatFieldFilePath, 'r')
            self.flatField_output = numpy.array(f['data']['value'])  # f['data0']
            f.close()
            self.flatField_input = numpy.array(self.flatField_input)
            if self.output_channels==1:
                #self.flatField_output = numpy.sum(self.flatField_output,2)
                self.flatField_input = numpy.mean(self.flatField_output, 2)

    # THIS IS THE INTERNAL FUNCTION THAT ACTUALLY LOADS THE DATA FROM FILE #
    def _initCache_locked_(self):
        startId = 0
        loadData_flag = True
        #wait_flag = True
        input_target_size = self.target_size + (self.input_channels,)
        output_target_size = self.target_size + (self.output_channels,)

        with self._lock_:
            startId = self.cacheRenewed_counter.value
            if(startId>=self.cache_size):
                loadData_flag = False
            else:
                loadData_flag=True
                self.cacheRenewed_counter.value+=self.batch_size
        
        if loadData_flag == True:
            # ---------------- #
            # repopulate cache #
            # ---------------- #
            idxArray = numpy.random.randint(0, self.numImages, self.cache_size)
            for ii in itertools.islice(itertools.count(), startId, min([startId + self.batch_size, self.cache_size])):
                imgIndex = idxArray[ii]
                inName = self.fileArray[imgIndex]
                f = h5py.File(inName, 'r')
                imX = numpy.array(f['Data_X'], order='F').transpose()
                imY = numpy.array(f['Data_Y'], order='F').transpose()
                f.close()
                # now doing the resize here to also scale the channels
                imX = transform.resize(imX, input_target_size, order=3, mode='reflect')
                imY = transform.resize(imY, output_target_size, order=3, mode='reflect')

                if len(imX.shape) < 3:
                    imX = imX.reshape(imX.shape + (1,))
                if len(imY.shape) < 3:
                    imY = imY.reshape(imY.shape + (1,))


                #if imX.shape != imY.shape:
                #    raise RuntimeError("Input- and Output sizes do not match.")

                # == Note: do data normalization here to reduce memory footprint ==#
                """
                Data Normalisation
                """
                if self.useNormData:
                    minValX, maxValX, imX = normaliseFieldArray(imX, self.input_channels, self.flatField_input, self.x_type)
                    minValY, maxValY, imY = normaliseFieldArray(imY, self.output_channels, self.flatField_output, self.y_type)

                imX = imX.astype(numpy.float32)
                imY = imY.astype(numpy.float32)
                with self._memlock_:
                    self.cacheX[ii] = imX
                    self.cacheY[ii] = imY
        else:
            return

        with self._lock_:
            if self.cacheRenewed_counter.value >= self.cache_size:
                self.cacheRenewed_counter.value = 0

            if self.cacheUsed_counter.value >= self.cache_period:
                self.cacheUsed_counter.value=0

            if((startId+self.batch_size)>=self.cache_size):
                self.renew_cache.value=False
        return


    def __initCache_open_(self):
        input_target_size = self.target_size + (self.input_channels,)
        output_target_size = self.target_size + (self.output_channels,)

        # SORRY BUT: WHAT THE FUCK IS THIS ?!
        # YOU HAVE A CACHE AND THEN DECIDE TO RENEW THE FULL CACHE AT EACH STEP ???
        # THIS IS SURELY NOT THE CODE I GAVE YOU
        if self.useCache:
            # ---------------- #
            # repopulate cache #
            # ---------------- #
            idxArray = numpy.random.randint(0, self.numImages, self.cache_size)
            for ii in itertools.islice(itertools.count(), 0, self.cache_size):
                imgIndex = idxArray[ii]
                inName = self.fileArray[imgIndex]
                f = h5py.File(inName, 'r')
                imX = numpy.array(f['Data_X'], order='F').transpose()
                imY = numpy.array(f['Data_Y'], order='F').transpose()
                f.close()
                # now doing the resize here to also scale the channels
                imX = transform.resize(imX, input_target_size, order=3, mode='reflect')
                imY = transform.resize(imY, output_target_size, order=3, mode='reflect')
                if len(imX.shape) < 3:
                    imX = imX.reshape(imX.shape + (1,))
                if len(imY.shape) < 3:
                    imY = imY.reshape(imY.shape + (1,))


                #if imX.shape != imY.shape:
                #    raise RuntimeError("Input- and Output sizes do not match.")

                # == Note: do data normalization here to reduce memory footprint ==#
                """
                Data Normalisation
                """
                if self.useNormData:
                    minValX, maxValX, imX = normaliseFieldArray(imX, self.input_channels, self.flatField_input, self.x_type)
                    minValY, maxValY, imY = normaliseFieldArray(imY, self.output_channels, self.flatField_output, self.y_type)

                imX = imX.astype(numpy.float32)
                imY = imY.astype(numpy.float32)
                with self._memlock_:
                    self.cacheX[ii] = imX
                    self.cacheY[ii] = imY

    def __len__(self):
        return int(numpy.ceil(len(self.fileArray)/float(self.batch_size)))
    
    def __getitem__(self, idx):
        if self.useCache:
            flushCache = False
            with self._lock_:
                flushCache = self.renew_cache.value
            if flushCache == True:
                self._initCache_locked_()

        batchX = numpy.zeros(self.batch_image_size_X, dtype=numpy.float32)
        batchY = numpy.zeros(self.batch_image_size_Y, dtype=numpy.float32)
        idxArray = numpy.random.randint(0, self.cache_size, self.batch_size)
        img_per_batch = max(1,int(self.batch_size / self.input_channels))
        for j in itertools.islice(itertools.count(),0,img_per_batch):
            outImgX = None
            outImgY = None
            if self.useCache:
                #imgIndex = numpy.random.randint(0, self.cache_size)
                imgIndex =idxArray[j]
                with self._memlock_:
                    outImgX = self.cacheX[imgIndex]
                    outImgY = self.cacheY[imgIndex]
            else:
                # imgIndex = min([(idx*self.batch_size)+j, self.numImages-1,len(self.inputFileArray)-1])
                imgIndex = ((idx * self.batch_size) + j) % (self.numImages - 1)
                """
                Load data from disk
                """
                inName = self.fileArray[imgIndex]
                f = h5py.File(inName, 'r')
                outImgX = numpy.array(f['Data_X'], order='F').transpose()
                outImgY = numpy.array(f['Data_Y'], order='F').transpose()
                f.close()
                if len(outImgX.shape) < 3:
                    outImgX = outImgX.reshape(outImgX.shape + (1,))
                if len(outImgY.shape) < 3:
                    outImgY = outImgY.reshape(outImgY.shape + (1,))                
                if outImgX.shape != outImgY.shape:
                    raise RuntimeError("Input- and Output sizes do not match.")
                # == Note: do data normalization here to reduce memory footprint ==#
                """
                Data Normalisation
                """
                if self.useNormData:
                    minValX,maxValX,outImgX = normaliseFieldArray(outImgX, self.input_channels, self.flatField_input, self.x_type)
                    minValY,minValY,outImgY = normaliseFieldArray(outImgY, self.output_channels, self.flatField_output, self.y_type)
                    outImgX = outImgX.astype(numpy.float32)
                    outImgY = outImgY.astype(numpy.float32)
               
            input_target_size = self.target_size + (self.input_channels, )
            output_target_size = self.target_size + (self.output_channels, ) 
            """
            Data augmentation
            """
            if self.useZoom:
                if self.useZoom:
                    outImgX = ndimage.zoom(outImgX, [self.zoomFactor, self.zoomFactor, 1], order=3, mode='reflect')
                    outImgX = outImgX[self.im_bounds[0]:self.im_bounds[1], self.im_bounds[2]:self.im_bounds[3],
                              0:self.input_channels]
                    outImgY = ndimage.zoom(outImgY, [self.zoomFactor, self.zoomFactor, 1], order=3, mode='constant')
                    outImgY = outImgY[self.im_bounds[0]:self.im_bounds[1], self.im_bounds[2]:self.im_bounds[3],
                              0:self.output_channels]
            #if self.useResize:
            #    outImgX = transform.resize(outImgX, input_target_size, order=3, mode='reflect')
            #    outImgY = transform.resize(outImgY, output_target_size, order=3, mode='reflect')
            if self.useCrop:
                outImgX = outImgX[self.im_bounds[0]:self.im_bounds[1], self.im_bounds[2]:self.im_bounds[3],
                          0:self.input_channels]
                outImgY = outImgY[self.im_bounds[0]:self.im_bounds[1], self.im_bounds[2]:self.im_bounds[3],
                          0:self.output_channels]
            if self.useFlipping:
                mode = numpy.random.randint(0,4)
                if mode == 0:   # no modification
                    pass
                if mode == 1:
                    outImgX = numpy.fliplr(outImgX)
                    outImgY = numpy.fliplr(outImgY)
                    #outImgX = ndimage.geometric_transform(outImgX, flipSpectralX, mode='nearest')
                    #outImgY = ndimage.geometric_transform(outImgY, flipSpectralX, mode='nearest')
                if mode == 2:
                    outImgX = numpy.flipud(outImgX)
                    outImgY = numpy.flipud(outImgY)
                    #outImgX = ndimage.geometric_transform(outImgX, flipSpectralY, mode='nearest')
                    #outImgY = ndimage.geometric_transform(outImgY, flipSpectralY, mode='nearest')
                if mode == 3:
                    outImgX = numpy.fliplr(outImgX)
                    outImgX = numpy.flipud(outImgX)
                    outImgY = numpy.fliplr(outImgY)
                    outImgY = numpy.flipud(outImgY)
                    #outImgX = ndimage.geometric_transform(outImgX, flipSpectralXY, mode='nearest')
                    #outImgY = ndimage.geometric_transform(outImgY, flipSpectralXY, mode='nearest')
            if self.useAWGN: # only applies to input data
                if self.input_channels > 1:
                    for channelIdx in itertools.islice(itertools.count(), 0, self.input_channels):
                        rectMin = numpy.min(outImgX[:,:,channelIdx])
                        rectMax = numpy.max(outImgX[:,:,channelIdx])
                        outImgX[:,:,channelIdx] = util.random_noise(outImgX[:,:,channelIdx], mode='gaussian', mean=self.MECTnoise_mu[channelIdx]*0.15, var=(self.MECTnoise_sigma[channelIdx]*0.15*self.MECTnoise_sigma[channelIdx]*0.15))
                        outImgX[:,:,channelIdx] = numpy.clip(outImgX[:,:,channelIdx], rectMin, rectMax)
                else:
                    rectMin = numpy.min(outImgX[:, :, 0])
                    rectMax = numpy.max(outImgX[:, :, 0])
                    outImgX[:, :, channelIdx] = util.random_noise(outImgX[:, :, 0], mode='gaussian',
                                                mean=self.SECTnoise_mu * 0.15, var=(self.SECTnoise_sigma * 0.15 * self.SECTnoise_sigma * 0.15))
                    outImgX[:, :, channelIdx] = numpy.clip(outImgX[:, :, channelIdx], rectMin, rectMax)
            if self.useMedian:
                mSize = self.medianSize[numpy.random.randint(0,len(self.medianSize))]
                if mSize > 0:
                    outImgX = ndimage.median_filter(outImgX, (mSize, mSize, 1), mode='constant', cval=1.0)

            if self.useGaussian:
                # here, it's perhaps incorrect to also smoothen the output;
                # rationale: even an overly smooth image should result is sharp outputs
                sigma = numpy.random.uniform(low=self.gaussianRange[0], high=self.gaussianRange[1])
                outImgX = ndimage.gaussian_filter(outImgX, (sigma, sigma, 0))
                #outImgY = ndimage.gaussian_filter(outImgY, (sigma, sigma, 0))
            """
            Store data if requested
            """
            if self.save_to_dir != None:
                sXImg = array_to_img(outImgX[:,:,0])
                #save_img(os.path.join(self.save_to_dir,fname_in+"."+self.save_format),sXimg)
                fname_in = "img_"+str(imgIndex)
                sXImg.save(os.path.join(self.save_to_dir,fname_in+"."+self.save_format))
                sYImg = array_to_img(outImgY[:,:,0])
                #save_img(os.path.join(self.save_to_dir,fname_out+"."+self.save_format), sYImg)
                fname_out = "img_" + str(imgIndex)
                sYImg.save(os.path.join(self.save_to_dir,fname_out+"."+self.save_format))
            for jj in itertools.islice(itertools.count(), 0, self.input_channels):
                batchX[(j*self.input_channels)+jj,:,:,0] = outImgX[:,:,jj]
                batchY[(j*self.input_channels)+jj,:,:,0] = outImgY[:,:,jj]
            if self.useCache:
                self._lock_.acquire()
                self.cacheUsed_counter.value+=1
                if int(self.cacheUsed_counter.value) >= self.cache_period:
                    self.renew_cache.value = True
                self._lock_.release()
        return batchX, batchY
    
    def on_epoch_end(self):
        self.__initCache_open_()
        

#class ScatterPhantomGenerator_inMemory(Sequence):
#   
#    def __init__(self, images_in, images_out, image_size=(128, 128,1), batch_size=1, useAWGN = False, gauss_mu=0.5, gauss_stdDev=0.1, useRotation=False, rotationRange=(0,360), targetSize=(128,128), useZoom=False, zoomFactor=1.0, useFlipping=False, useNormData=False, save_to_dir=None, save_format="png"):
#        self.batch_size = batch_size
#        self.image_size = image_size
#        #self.image_path = image_path
#        #self.augment_flow = augment_flow
#        #self.k = (self.kernel_size-1)//2
#        self.dtype = numpy.float32
#        self.useAWGH = useAWGN
#        self.gauss_mu = gauss_mu
#        self.gauss_stdDev = gauss_stdDev
#        self.rotationRange = rotationRange
#        self.targetSize=targetSize
#        self.useRotation = useRotation
#        self.useClip = useRotation
#        self.zoomFactor = zoomFactor
#        self.useZoom = useZoom
#        self.useFlipping = useFlipping
#        self.useNormData = useNormData
#        dims = targetSize
#        self.targetSize = (self.targetSize[0], self.targetSize[1], self.image_size[2])
#        #========================================#
#        #== clipping-related image information ==#
#        #========================================#
#        self.im_center = numpy.array([(self.image_size[0]-1)/2, (self.image_size[1]-1)/2], dtype=numpy.int32)
#        self.im_size = numpy.array([(self.targetSize[0]-1)/2, (self.targetSize[1]-1)/2], dtype=numpy.int32)
#        left = self.im_center[0]-self.im_size[0]
#        right = left+self.targetSize[0]
#        top = self.im_center[1]-self.im_size[1]
#        bottom = top+self.targetSize[1]
#        self.im_bounds = (left,right,top,bottom)
#        #===================================#
#        #== directory-related information ==#
#        #===================================#
#        self.X = images_in
#        self.Y = images_out
#        self.numImages = self.X.shape[0]
#        outImgX = self.X[0]
#        if len(outImgX)<3:
#            outImgX = outImgX.reshape(outImgX.shape + (1,))
#        self.image_size =outImgX.shape
#        self.save_to_dir=save_to_dir
#        self.save_format=save_format
#    
#    def __len__(self):
#        return int(numpy.ceil(self.numImages/float(self.batch_size)))
#    
#    def __getitem__(self, idx):
#        batchX = numpy.zeros((self.batch_size,self.image_size[0],self.image_size[1],self.image_size[2]),dtype=self.dtype)
#        batchY = numpy.zeros((self.batch_size,self.image_size[0],self.image_size[1],self.image_size[2]),dtype=self.dtype)
#        if self.useClip or self.useZoom:
#            batchX = numpy.zeros((self.batch_size,self.targetSize[0],self.targetSize[1],self.image_size[2]),dtype=self.dtype)
#            batchY = numpy.zeros((self.batch_size,self.targetSize[0],self.targetSize[1],self.image_size[2]),dtype=self.dtype)
#        for j in itertools.islice(itertools.count(),0,self.batch_size):
#            #imgIndex = min([(idx*self.batch_size)+j, self.numImages-1])
#            imgIndex = ((idx*self.batch_size)+j) % (self.numImages-1)
#            #if shuffle:
#            #    batchIndex = numpy.random.randint(0, min([self.numImages,len(self.inputFileArray)]))
#            """
#            Load data from memory
#            """
#            outImgX = self.X[imgIndex]
#            outImgY = self.Y[imgIndex]
#            if len(outImgX)<3:
#                outImgX = outImgX.reshape(outImgX.shape + (1,))
#            if len(outImgY)<3:
#                outImgY = outImgY.reshape(outImgY.shape + (1,))
#            if outImgX.shape != outImgY.shape:
#                raise RuntimeError("Input- and Output sizes do not match.")
#            #self.image_size =outImgX.shape
#            """
#            Data augmentation
#            """
#            if self.useNormData:
#                minValX = numpy.min(outImgX)
#                maxValX = numpy.max(outImgX)
#                outImgX = (outImgX-minValX)/(maxValX-minValX)
#                outImgX = outImgX.astype(numpy.float32)
#                minValY = numpy.min(outImgY)
#                maxValY = numpy.max(outImgY)
#                outImgY = (outImgY-minValY)/(maxValY-minValY)
#                outImgY = outImgY.astype(numpy.float32)
#            if self.useZoom:
#                outImgX = ndimage.zoom(outImgX, [self.zoomFactor,self.zoomFactor,1], order=3)
#                outImgY = ndimage.zoom(outImgY, [self.zoomFactor,self.zoomFactor,1], order=3)
#            if self.useRotation:
#                outImgX = ndimage.rotate(outImgX, numpy.random.uniform(self.rotationRange[0], self.rotationRange[1]), axes=(1,0), order=2, mode='mirror')
#                outImgY = ndimage.rotate(outImgY, numpy.random.uniform(self.rotationRange[0], self.rotationRange[1]), axes=(1,0), order=2, mode='mirror')
#            if self.useClip:
#                outImgX = outImgX[self.im_bounds[0]:self.im_bounds[1],self.im_bounds[2]:self.im_bounds[3],0:self.targetSize[2]]
#                outImgY = outImgY[self.im_bounds[0]:self.im_bounds[1],self.im_bounds[2]:self.im_bounds[3],0:self.targetSize[2]]
#            if self.useAWGH: # only applies to input data
#                outImgX = util.random_noise(outImgX, mode='gaussian', mean=self.gauss_mu, var=(self.gauss_stdDev*self.gauss_stdDev))
#            if self.useFlipping:
#                mode = numpy.random.randint(0,4)
#                if mode == 0:   # no modification
#                    pass
#                if mode == 1:
#                    outImgX = numpy.fliplr(outImgX)
#                    outImgY = numpy.fliplr(outImgY)
#                    #outImgX = ndimage.geometric_transform(outImgX, flipSpectralX, mode='nearest')
#                    #outImgY = ndimage.geometric_transform(outImgY, flipSpectralX, mode='nearest')
#                if mode == 2:
#                    outImgX = numpy.flipud(outImgX)
#                    outImgY = numpy.flipud(outImgY)
#                    #outImgX = ndimage.geometric_transform(outImgX, flipSpectralY, mode='nearest')
#                    #outImgY = ndimage.geometric_transform(outImgY, flipSpectralY, mode='nearest')
#                if mode == 3:
#                    outImgX = numpy.fliplr(outImgX)
#                    outImgX = numpy.flipud(outImgX)
#                    outImgY = numpy.fliplr(outImgY)
#                    outImgY = numpy.flipud(outImgY)
#                    #outImgX = ndimage.geometric_transform(outImgX, flipSpectralXY, mode='nearest')
#                    #outImgY = ndimage.geometric_transform(outImgY, flipSpectralXY, mode='nearest')
#            """
#            Store data if requested
#            """
#            if self.save_to_dir != None:
#                sXImg = array_to_img(outImgX[:,:,0])
#                #save_img(os.path.join(self.save_to_dir,fname_in+"."+self.save_format),sXimg)
#                sXimg.save(os.path.join(self.save_to_dir,fname_in+"."+self.save_format))
#                sYImg = array_to_img(outImgY[:,:,0])
#                #save_img(os.path.join(self.save_to_dir,fname_out+"."+self.save_format), sYImg)
#                sYimg.save(os.path.join(self.save_to_dir,fname_out+"."+self.save_format))
#            batchX[j] = outImgX
#            batchY[j] = outImgY
#        return batchX, batchY
