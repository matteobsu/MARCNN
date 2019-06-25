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

class ScatterPhantomGenerator(Sequence):
    
    def __init__(self, batch_size=1, image_size=(128, 128), input_channels=32, target_size=(128, 128), output_channels=1, useResize=False,
                 useCrop=False, useZoom=False, zoom_factor_range=(0.95,1.05), useAWGN = False, useMedian=False, useGaussian=False,
                 useFlipping=False, useNormData=False, cache=None, save_to_dir=None, save_format="png", threadLockVar=None, useCache=False):
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
#                        self.cache.set_item_limits_x(indices[index], minValX, maxValX)
                        self.cache.set_cache_item_y(indices[index],imY_slice)
#                        self.cache.set_item_limits_y(indices[index], minValY, maxValY)

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
                imX = imX.astype(numpy.float32)
                imY = imY.astype(numpy.float32)
                with self._lock_:
                    self.cache.set_cache_item_x(ii,imX)
#                    self.cache.set_item_limits_x(ii, minValX, maxValX)
                    self.cache.set_cache_item_y(ii, imY)
#                    self.cache.set_item_limits_y(ii, minValY, maxValY)

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
                imX = imX.astype(numpy.float32)
                imY = imY.astype(numpy.float32)

            fname_in = "img_{}_{}_{}".format(self._epoch_num_, idx, j)
            """
            Data augmentation
            """
            input_target_size = self.target_size + (self.input_channels, )
            output_target_size = self.target_size + (self.output_channels, )
#            if self.useZoom:
#                _zoom_factor = numpy.random.uniform(self.zoom_factor_range[0], self.zoom_factor_range[1])
#                for channelIdx in itertools.islice(itertools.count(), 0, self.input_channels):
#                    imX[:,:,channelIdx] = clipped_zoom(imX[:,:,channelIdx], _zoom_factor, order=3, mode='reflect')
#                    imY[:,:,channelIdx] = clipped_zoom(imY[:,:,channelIdx], _zoom_factor, order=3, mode='reflect')
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