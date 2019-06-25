#!python3
"""
Modified UNet training for scatter estimation 

Created on Jul 26, 2018

@author: christian (Christian Kehl)
@mail: chke@dtu.dk
"""

from __future__ import print_function
from UNet_MCCNN_MECT import UNetFactory
from ScatteringPhantom_externalMem import ScatterPhantomGenerator
import MemCache_SCATTER

from argparse import ArgumentParser
import os
#from scipy import io
#import h5py
import pickle
import keras
#import numpy
import multiprocessing
#from multiprocessing.managers import BaseManager
from callbacks import ModelCheckpoint
import math

#import threading


"""
General parameters for learning
"""
input_indicator = "*X*"
output_indicator = "*Y*"
epochs = 20
keep_period = 5
input_image_size = (128,128)       # needs to fit with the actual data dim
target_image_size = (96,96)  # free to choose for resizing
input_image_shape = (128,128,32,1)
target_image_shape = (96,96,32,1)
epoch_steps = 402
input_channels = 32
output_channels = 32
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
                               action="store", dest="train-dir",
                               default="../data/train", help=tdi_help)
    
    vdi_help = "validation input data directory"
    option_parser.add_argument("-V", "--valid-dir-in",
                               action="store", dest="valid-dir",
                               default="../data/train", help=vdi_help)
    
    aug_help = ("augment input data")
    option_parser.add_argument("--augment-input", action="store_true",
                               dest="augment", help=aug_help)
    
    #c_help = ("name of the continuing, previous model")
    #option_parser.add_argument("-c", "--continue_model", action="store", dest="continue_mfile", default="", help=c_help)
    
    options = option_parser.parse_args()
    arg_dict = vars(options)
    
    training_directory = os.path.join(arg_dict["train-dir"])
    validation_directory = os.path.join(arg_dict["valid-dir"])
    resultpath = arg_dict["resultpath"]
    
    if not os.path.exists(os.path.join(resultpath, "output")):
        os.makedirs(os.path.join(resultpath, "output"))
    if not os.path.exists(os.path.join(resultpath, "models")):
        os.makedirs(os.path.join(resultpath, "models"))
    
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
##
#    manager = Keras_Concurrency()
#    manager.register('MemoryCache', MemCache_SCATTER.MemoryCache)
#    manager.start()
#    
#    cache_mem = manager.MemoryCache()
#    valid_cache_mem = manager.MemoryCache()
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
        datagen = ScatterPhantomGenerator(batchSize,input_image_size,input_channels,target_image_size,output_channels,
                                          useResize=True,useCrop=False,useZoom=False,useAWGN=True,useMedian=True,useGaussian=False,
                                          useFlipping=True,useNormData=True, threadLockVar=trainingLock,useCache=True,cache=cache_mem)
        datagen.prepareDirectFileInput([training_directory])
        nsteps = int(math.ceil(epoch_steps / batchSize))
        datagen.set_nsteps(nsteps)

        validgen = ScatterPhantomGenerator(batchSize,input_image_size,input_channels,target_image_size,output_channels,
                                          useResize=True,useCrop=False,useZoom=False,useAWGN=False,useMedian=False,useGaussian=False,
                                          useFlipping=False,useNormData=True, threadLockVar=trainingLock,useCache=True,cache=valid_cache_mem)
        validgen.prepareDirectFileInput([validation_directory])
        valid_nsteps = min([int(validgen.numImages / batchSize), int(128 / batchSize)])
        validgen.set_nsteps(valid_nsteps)
    else:      
        datagen = ScatterPhantomGenerator(batchSize,input_image_size,input_channels,target_image_size,output_channels,
                                          useResize=True,useCrop=False,useZoom=False,useAWGN=False,useMedian=False,useGaussian=False,
                                          useFlipping=False,useNormData=True, threadLockVar=trainingLock,useCache=True,cache=cache_mem,save_to_dir="D:\\mbusi\\SCNN\\dumpfold\\train\\")
        datagen.prepareDirectFileInput([training_directory])
        nsteps = int(math.ceil(epoch_steps / batchSize))
        datagen.set_nsteps(nsteps)
        
        validgen = ScatterPhantomGenerator(batchSize,input_image_size,input_channels,target_image_size,output_channels,
                                          useResize=True,useCrop=False,useZoom=False,useAWGN=False,useMedian=False,useGaussian=False,
                                          useFlipping=False,useNormData=True, threadLockVar=trainingLock,useCache=True,cache=valid_cache_mem,save_to_dir="D:\\mbusi\\SCNN\\dumpfold\\valid\\")
        validgen.prepareDirectFileInput([validation_directory])
        valid_nsteps = min([int(validgen.numImages / batchSize), int(128 / batchSize)])
        validgen.set_nsteps(valid_nsteps)
    
    callbacks = []
    callbacks.append(ModelCheckpoint(modelpath=modelfile,
                                    weightspath=weightfile,
                                    period=1,
                                    auto_remove_model=True,
                                    auto_remove_weights=True,
                                    keep_period=keep_period))
    
# UBUNTU SAFE
# =============================================================================
#    import subprocess
#    subprocess.Popen("timeout 200 nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1 \ sed s/%//g > ./GPU_stats.log",shell=True)
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
                                                steps_per_epoch=int(datagen.numImages/batchSize),
#                                                steps_per_epoch=int(datagen.numImages/batchSize),
#                                                validation_steps=int(validgen.numImages/batchSize),
                                                validation_data=validgen,
                                                validation_steps=min([int(validgen.numImages/batchSize), int(128/batchSize)]),
                                                epochs=epochs,
                                                use_multiprocessing=False,
#                                                workers=12,
                                                callbacks=callbacks)

    with open(mhistfile.format(epoch=epochs), 'wb') as file:
        pickle.dump(model_fitting_history.history, file)
