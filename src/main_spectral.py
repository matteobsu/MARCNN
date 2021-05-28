#!python3
"""
Modified UNet training for scatter estimation 

Created on Jul 26, 2018
s
"""

from __future__ import print_function
from UNet_MCCNN_MECT import UNetFactory
#from UNet2D_Maier2018_spectral import UNet2D_Maier2018_spectral
#from UNet2D_Maier2018 import UNet2D_Maier2018

#from ScatteringPhantom import ScatterPhantomGenerator
from ScatteringPhantom_externalMem_NN import ScatterPhantomGenerator
#from ScatteringPhantom_spectral import ScatterPhantomGenerator
import MemCache_SCATTER

from argparse import ArgumentParser
import os
from scipy import io
import h5py
import pickle
import keras
import numpy
import multiprocessing
from multiprocessing.managers import BaseManager
from callbacks import ModelCheckpoint

import threading
import multiprocessing


"""
General parameters for learning
"""
input_indicator = "*X*"
output_indicator = "*Y*"
epochs = 50
keep_period = 5
#image_shape = (256,256,1)       # needs to fit with the actual data dim
#image_shape = (256,256,32)       # needs to fit with the actual data dim
#image_shape = (150,150,32)       # needs to fit with the actual data dim
#image_shape = (437,437,1)       # needs to fit with the actual data dim
#image_shape = (437,437,32)       # needs to fit with the actual data dim
#targetImageShape = (256,256,1)  # free to choose for resizing
#targetImageShape = (256,256,32)  # free to choose for resizing
#targetImageShape = (64,64,8)  # free to choose for resizing
input_image_size = (256,256)       # needs to fit with the actual data dim
target_image_size = (256,256)  # free to choose for resizing
input_image_shape = (256,256,8,1)
target_image_shape = (256,256,8,1)
input_channels = 8
output_channels = 8
epoch_steps=4096
train_samples_n=466
validation_samples_n=110
batchSize=8

if __name__ == '__main__':
    description = ("Creating model (modified U-Net) and training routine for CT " +
                   "denoise DeepLearning\n\n" +
                   "Notice!! The training and validation now needs separate " +
                   "pickle files. See arguments: -t, -T, -v, and -V.")
    option_parser = ArgumentParser(description=description)
    
    r_help = ("path to data folder with results (i.e. parent of output- and " +
              "model folder)")
    option_parser.add_argument("-r", "--resultpath",
                               action="store", dest="resultpath",
                               default="../data", help=r_help)
    m_help = ("name of the model file")
    option_parser.add_argument("-m", "--modelname",
                               action="store", dest="modelname",
                               default="reCT_epoch_{epoch:04d}", help=m_help)
    
    tdi_help = "training input data directory"
    option_parser.add_argument("-T", "--train-dir-in",
                               action="store", dest="train-dir-in",
                               default="../data/train", help=tdi_help)
    tdo_help = "training output data directory"
    option_parser.add_argument("-t", "--train-dir-out",
                               action="store", dest="train-dir-out",
                               default="../data/train", help=tdo_help)
    vdi_help = "validation input data directory"
    option_parser.add_argument("-V", "--valid-dir-in",
                               action="store", dest="valid-dir-in",
                               default="../data/train", help=vdi_help)
    vdo_help = "validation output data directory"
    option_parser.add_argument("-v", "--valid-dir-out",
                               action="store", dest="valid-dir-out",
                               default="../data/train", help=vdo_help)

    ff_help = "flat field file path"
    option_parser.add_argument("-F", "--flatfield-path",
                               action="store", dest="flatfield-path",
                               default=None, help=ff_help)
    aug_help = ("augment input data")
    option_parser.add_argument("--augment-input", action="store_true",
                               dest="augment", help=aug_help)
    
    #c_help = ("name of the continuing, previous model")
    #option_parser.add_argument("-c", "--continue_model", action="store", dest="continue_mfile", default="", help=c_help)
    
    options = option_parser.parse_args()
    arg_dict = vars(options)
    
    training_directory_X = os.path.join(arg_dict["train-dir-in"])
    training_directory_Y = os.path.join(arg_dict["train-dir-out"])
    validation_directory_X = os.path.join(arg_dict["valid-dir-in"])
    validation_directory_Y = os.path.join(arg_dict["valid-dir-out"])
    resultpath = arg_dict["resultpath"]
    
    if not os.path.exists(os.path.join(resultpath, "output")):
        os.makedirs(os.path.join(resultpath, "output"))
    if not os.path.exists(os.path.join(resultpath, "models")):
        os.makedirs(os.path.join(resultpath, "models"))
        
    flatfield_path = None
    if(arg_dict["flatfield-path"]!=None):
        flatfield_path = arg_dict["flatfield-path"]
    
    modelfilename = arg_dict["modelname"]
    mhistfile = os.path.join(resultpath, "output", modelfilename + "_Thist.pkl")
    weightfile = os.path.join(resultpath, "models", modelfilename + "_weights.h5")
    modelfile = os.path.join(resultpath, "models", modelfilename + "_model.h5")
    
    # ============ KEEP THIS COMMENTED - USEFUL FOR SUBSEQUENT CHAINED TRAINING ============= #
    #cont_modelfile = os.path.join(resultpath, "models", modelfilename + "_model.h5")
    #cont_weightfile = os.path.join(resultpath, "models", modelfilename + "_weights.h5")
    #if len(arg_dict["continue_mfile"]) > 0:
    #    cont_modelfile = os.path.join(resultpath, "models", arg_dict["continue_mfile"] + "_model.h5")
    #    cont_weightfile = os.path.join(resultpath, "models", arg_dict["continue_mfile"] + "_weights.h5")
    #    last = ModelCheckpoint.last_checkpoint_epoch_and_model(cont_modelfile)
    #    latest_epoch, latest_model = last
    
    latest_epoch, latest_model = 0, None
    ModelCheckpoint.remove_all_checkpoints(modelfile, weightfile)

    """
    Our UNet model is defined in the module unetwork stored locally. Using a
    factory class we can easyly generate a custom model using very few steps.
    
    unet_factory.dropout = None
    unet_factory.convolution_kernel_size = 3
    unet_factory.batch_normalization = False
    unet_factory.begin(image_shape=image_shape)
    unet_factory.generateLevels(init_filter_count=32, recursive_depth=4)
    model = unet_factory.finalize(final_filter_count=1)
    
    Here, using the specific network of Maier et al. 2018 on Deep Scatter Estimation (DSE)
    """
    #unet = UNet2D_Maier2018_spectral()
    ##unet = UNet2D_Maier2018()
    #unet.begin((target_image_size[0],target_image_size[1],input_channels),(target_image_size[0],target_image_size[1],output_channels))
    #unet.buildNetwork()
    #model = unet.finalize()
    nnet = UNetFactory()
    nnet.begin(image_shape=target_image_shape)
    model = nnet.buildNetwork()

    """
    Adding an optimizer (optional)---can be used to optimize gradient decent.
    We employ the rmsprop: divide the gradient by a running average of its
    recent magnitude.
    These are, for the moment, arbitrarily chosen by the original author. The
    work would benifit from a thorough walk through these choices.
    """
    #opt = keras.optimizers.rmsprop(lr=0.0001,  # Learning rate (lr)
    #                               decay=1e-6)  # lr decay over each update.
    #opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6, amsgrad=False)
    opt = keras.optimizers.Adam(lr=0.0001, decay=3.0e-07, amsgrad=True)
    model.compile(loss='mean_absolute_error',  # standard cost function
                  optimizer=opt,
                  metrics=['mse'])  # Mean Absolute Error metrics

    """
    Data augmentation setup
    """    
    augment = arg_dict["augment"]
    trainingLock = multiprocessing.Lock()
    validationLock = multiprocessing.Lock()

# UBUNTU SAFE
# =============================================================================
#    class Keras_Concurrency(BaseManager):
#        pass
#
#    manager = Keras_Concurrency()
#    manager.register('MemoryCache', MemCache_SCATTER.MemoryCache)
#    manager.start()
#    cache_mem = manager.MemoryCache()
#    cache_mem.set_image_shape_x(input_image_shape)
#    cache_mem.set_number_channels_x(output_channels)
#    cache_mem.set_image_shape_y(input_image_shape)
#    cache_mem.set_number_channels_y(output_channels)
#    cache_mem.allocate()
#
#    valid_cache_mem = manager.MemoryCache()
#    valid_cache_mem.set_image_shape_x(input_image_shape)
#    valid_cache_mem.set_number_channels_x(output_channels)
#    valid_cache_mem.set_image_shape_y(input_image_shape)
#    valid_cache_mem.set_number_channels_y(output_channels)
#    valid_cache_mem.allocate()
# =============================================================================
    cache_mem = MemCache_SCATTER.MemoryCache()
    cache_mem.set_image_shape_x(input_image_shape)
    cache_mem.set_number_channels_x(output_channels)
    cache_mem.set_image_shape_y(input_image_shape)
    cache_mem.set_number_channels_y(output_channels)
    cache_mem.allocate()

    valid_cache_mem = MemCache_SCATTER.MemoryCache()
    valid_cache_mem.set_image_shape_x(input_image_shape)
    valid_cache_mem.set_number_channels_x(output_channels)
    valid_cache_mem.set_image_shape_y(input_image_shape)
    valid_cache_mem.set_number_channels_y(output_channels)
    valid_cache_mem.allocate()
# =============================================================================

    if augment:
        ##datagen = DataGenerator(image_shape, useAWGN=True, 0.05, 0.025, useRotation=False, rotationRange=(0,360), targetSize=(targetImageShape[0],targetImageShape[1]), useZoom=False, zoomFactor=scaleFactor, useFlipping=True)
        #datagen = DataGenerator(image_shape, useAWGN=False, useRotation=False, targetSize=(targetImageShape[0],targetImageShape[1]), useZoom=False, useFlipping=True)
        ##datagen = DataGenerator(image_shape, useAWGN=False, useRotation=False, targetSize=(targetImageShape[0],targetImageShape[1]), useZoom=False, useFlipping=False)
        ##datagen = DataGenerator(image_shape, useAWGN=False, useRotation=True, rotationRange=(0,45), targetSize=(targetImageShape[0],targetImageShape[1]), useZoom=False, useFlipping=True)
        #datagen.prepareDirectFileInput([training_directory])

        # gauss_mu=0.05, gauss_stdDev=0.025,
        datagen = ScatterPhantomGenerator(batchSize,input_image_size,input_channels,target_image_size,output_channels,
                                          useResize=False,useCrop=False,useZoom=False,useAWGN=True,useMedian=True,useGaussian=False,
                                          useFlipping=True,useNormData=True, threadLockVar=trainingLock,useCache=True,cache=cache_mem)
        datagen.prepareDirectFileInput([training_directory_X], flatfield_path)

        #validgen = ScatterPhantomGenerator(image_shape, batchSize, useAWGN=False, useRotation=False, targetSize=(targetImageShape[0],targetImageShape[1]), useZoom=False, useFlipping=False, useCache=False)
        #validgen.prepareDirectFileInput([validation_directory])
        validgen = ScatterPhantomGenerator(batchSize,input_image_size,input_channels,target_image_size,output_channels,
                                          useResize=False,useCrop=False,useZoom=False,useAWGN=False,useMedian=False,useGaussian=False,
                                          useFlipping=False,useNormData=True, threadLockVar=trainingLock,useCache=True,cache=valid_cache_mem)
        validgen.prepareDirectFileInput([validation_directory_X], flatfield_path)
    else:
        #datagen = StraightDirectoryGenerator(image_shape)
        #datagen.prepareDirectFileInput([training_directory])
        datagen = ScatterPhantomGenerator(batchSize,input_image_size,input_channels,target_image_size,output_channels,
                                          useResize=False,useCrop=False,useZoom=False,useAWGN=False,useMedian=False,useGaussian=False,
                                          useFlipping=False,useNormData=True, threadLockVar=trainingLock,useCache=True,cache=cache_mem)
        datagen.prepareDirectFileInput([training_directory_X], flatfield_path)

        #validgen = ScatterPhantomGenerator(image_shape, batchSize, useAWGN=False, useRotation=False, targetSize=(targetImageShape[0],targetImageShape[1]), useZoom=False, useFlipping=False, useCache=False)
        #validgen.prepareDirectFileInput([validation_directory])
        validgen = ScatterPhantomGenerator(batchSize,input_image_size,input_channels,target_image_size,output_channels,
                                          useResize=False,useCrop=False,useZoom=False,useAWGN=False,useMedian=False,useGaussian=False,
                                          useFlipping=False,useNormData=True, threadLockVar=trainingLock,useCache=True,cache=valid_cache_mem)
        validgen.prepareDirectFileInput([validation_directory_X], flatfield_path)
    
    callbacks = []
    callbacks.append(ModelCheckpoint(modelpath=modelfile,
                                    weightspath=weightfile,
                                    period=1,
                                    auto_remove_model=True,
                                    auto_remove_weights=True,
                                    keep_period=keep_period))
    
# UBUNTU SAFE
# =============================================================================
# #    import subprocess
# #    subprocess.Popen("timeout 200 nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1 \ sed s/%//g > ./GPU_stats.log",shell=True)
# =============================================================================

    # explanation for multiprocessing and workers:
    # workers: Integer. Maximum number of processes to spin up when using
    #     process-based threading. If unspecified, workers will default to 1.
    #     If 0, will execute the generator on the main thread.
    # use_multiprocessing: Boolean. If True, use process-based threading.
    #     If unspecified, use_multiprocessing will default to False. Note that
    #     because this implementation relies on multiprocessing, you should not
    #     pass non-picklable arguments to the generator as they can't be passed
    #     easily to children processes.
    # fits the model on batches with real-time data augmentation:
    model_fitting_history = model.fit_generator(datagen,
                                                #steps_per_epoch=int(train_samples_n/batchSize),
                                                validation_data=validgen,
                                                #validation_steps=int(validation_samples_n/batchSize),
                                                epochs=epochs,
#                                                use_multiprocessing=False,
                                                use_multiprocessing=True,
                                                workers=12,
                                                callbacks=callbacks)

    with open(mhistfile.format(epoch=epochs), 'wb') as file:
        pickle.dump(model_fitting_history.history, file)
