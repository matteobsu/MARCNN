#!python3
"""
Modified UNet training for scatter estimation 

Created on Jul 26, 2018

@author: christian (Christian Kehl)
@mail: chke@dtu.dk
"""

from __future__ import print_function

import glob
import itertools
import math
import re

from UNet_MCCNN_MECT import UNetFactory
from ScatteringPhantom_externalMem import ScatterPhantomGenerator_inMemory
#import MemCache_SCATTER

from argparse import ArgumentParser
import os
#from scipy import io
import h5py
import pickle
import keras
import numpy
import multiprocessing
#from multiprocessing.managers import BaseManager
from callbacks import ModelCheckpoint

#import threading


"""
General parameters for learning
"""
input_indicator = "*X*"
output_indicator = "*Y*"
epochs = 50
keep_period = 5
input_image_size = (128,128)       # needs to fit with the actual data dim
target_image_size = (96,96)  # free to choose for resizing
input_image_shape = (128,128,32,1)
target_image_shape = (96,96,32,1)
epoch_steps = 402
input_channels = 32
output_channels = 32
batchSize=32

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

    """
    Fill file arrays from directory
    """
    train_imX_mat = numpy.array([1,])
    train_imY_mat = numpy.array([1, ])
    fileArray_train = []
    validate_imX_mat = numpy.array([1, ])
    validate_imY_mat = numpy.array([1, ])
    fileArray_validate = []

    for name in glob.glob(os.path.join(training_directory, '*.h5')):
        fileArray_train.append(name)
    digits = re.compile(r'(\d+)')
    def tokenize(filename):
        return tuple(int(token) if match else token for token, match in
                         ((fragment, digits.search(fragment)) for fragment in digits.split(filename)))
    # = Now you can sort your file names like so: =#
    fileArray_train.sort(key=tokenize)
    nImages_train = len(fileArray_train)
    inImgDims = (input_image_size[0], input_image_size[1], input_channels, 1)
    outImgDims = (input_image_size[0], input_image_size[1], output_channels, 1)
    for name in glob.glob(os.path.join(validation_directory, '*.h5')):
        fileArray_validate.append(name)
    digits = re.compile(r'(\d+)')
    def tokenize(filename):
        return tuple(int(token) if match else token for token, match in
                         ((fragment, digits.search(fragment)) for fragment in digits.split(filename)))
    # = Now you can sort your file names like so: =#
    fileArray_validate.sort(key=tokenize)
    nImages_validate = len(fileArray_validate)

    train_imX_mat = numpy.zeros((len(fileArray_train),input_image_size[0],input_image_size[1],output_channels,1))
    train_imY_mat = numpy.zeros((len(fileArray_train),input_image_size[0],input_image_size[1],output_channels,1))
    run_index = 0
    for index in itertools.islice(itertools.count(), 0, len(fileArray_train)):
        inName = fileArray_train[index]
        f = h5py.File(inName, 'r')
        imX = numpy.array(f['Data_X'], order='F').transpose()
        imY = numpy.array(f['Data_Y'], order='F').transpose()
        f.close()
        if len(imX.shape) != len(imY.shape):
            print("Image dimensions do not match - EXITING ...")
            exit(1)

        if len(imX.shape) > 3:
            for ii in itertools.islice(itertools.count(), 0, len(imX.shape[2])):
                slice_index = ii
                imX_slice = numpy.squeeze(imX[:, :, slice_index, :])
                # === we need to feed the data as 3D+1 channel data stack === #
                if len(imX_slice.shape) < 3:
                    imX_slice = imX_slice.reshape(imX_slice.shape + (1,))
                if len(imX_slice.shape) < 4:
                    imX_slice = imX_slice.reshape(imX_slice.shape + (1,))
                imY_slice = numpy.squeeze(imY[:, :, slice_index, :])
                if len(imY_slice.shape) < 3:
                    imY_slice = imY_slice.reshape(imY_slice.shape + (1,))
                if len(imY_slice.shape) < 4:
                    imY_slice = imY_slice.reshape(imY_slice.shape + (1,))
                # == Note: do data normalization here to reduce memory footprint ==#
                imX_slice = imX_slice.astype(numpy.float32)
                imY_slice = imY_slice.astype(numpy.float32)
                train_imX_mat[run_index] = imX_slice
                train_imY_mat[run_index] = imY_slice
                run_index += 1
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
            train_imX_mat[run_index] = imX
            train_imY_mat[run_index] = imY
            # == Note: do data normalization here to reduce memory footprint ==#
            run_index+=1          
    validate_imX_mat = numpy.zeros((len(fileArray_validate),input_image_size[0],input_image_size[1],output_channels,1))
    validate_imY_mat = numpy.zeros((len(fileArray_validate),input_image_size[0],input_image_size[1],output_channels,1))
    run_index = 0
    for index in itertools.islice(itertools.count(), 0, len(fileArray_validate)):
        inName = fileArray_validate[index]
        f = h5py.File(inName, 'r')
        imX = numpy.array(f['Data_X'], order='F').transpose()
        imY = numpy.array(f['Data_Y'], order='F').transpose()
        f.close()
        if len(imX.shape) != len(imY.shape):
            print("Image dimensions do not match - EXITING ...")
            exit(1)

        if len(imX.shape) > 3:
            for ii in itertools.islice(itertools.count(), 0, len(imX.shape[2])):
                slice_index = ii
                imX_slice = numpy.squeeze(imX[:, :, slice_index, :])
                # === we need to feed the data as 3D+1 channel data stack === #
                if len(imX_slice.shape) < 3:
                    imX_slice = imX_slice.reshape(imX_slice.shape + (1,))
                if len(imX_slice.shape) < 4:
                    imX_slice = imX_slice.reshape(imX_slice.shape + (1,))
                imY_slice = numpy.squeeze(imY[:, :, slice_index, :])
                if len(imY_slice.shape) < 3:
                    imY_slice = imY_slice.reshape(imY_slice.shape + (1,))
                if len(imY_slice.shape) < 4:
                    imY_slice = imY_slice.reshape(imY_slice.shape + (1,))
                # == Note: do data normalization here to reduce memory footprint ==#
                imX_slice = imX_slice.astype(numpy.float32)
                imY_slice = imY_slice.astype(numpy.float32)
                validate_imX_mat[run_index] = imX_slice
                validate_imY_mat[run_index] = imY_slice
                run_index += 1
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
            validate_imX_mat[run_index] = imX
            validate_imY_mat[run_index] = imY
            # == Note: do data normalization here to reduce memory footprint ==#
            run_index+=1
    """
    Data arrays are filled
    """




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
    
#    model.compile(loss='mean_absolute_error',  # standard cost function
#                  optimizer=opt,
#                  metrics=['mse'])  # Mean Absolute Error metrics

    #-----------------------------------------------------------------#
    #---- activate this function and comment out the model lines  ----#
    #---- above to activate logarithmic learning error rate.      ----#
    #-----------------------------------------------------------------#
    model.compile(loss='mean_squared_logarithmic_error', # cost function
                  optimizer=opt,
                  metrics=['mae','mse'])  # Mean Absolute Error & Mean Squared Error metrics

    """
    Data augmentation setup
    """    
    augment = arg_dict["augment"]
    trainingLock = multiprocessing.Lock()
    validationLock = multiprocessing.Lock()


    #if augment:
    #    datagen = ScatterPhantomGenerator(batchSize,input_image_size,input_channels,target_image_size,output_channels,
    #                                      useResize=True,useCrop=False,useZoom=False,useAWGN=True,useMedian=True,useGaussian=False,
    #                                      useFlipping=True,useNormData=True, threadLockVar=trainingLock,useCache=True,cache=cache_mem)
    #    datagen.prepareDirectFileInput([training_directory])
    #
    #    validgen = ScatterPhantomGenerator(batchSize,input_image_size,input_channels,target_image_size,output_channels,
    #                                      useResize=True,useCrop=False,useZoom=False,useAWGN=False,useMedian=False,useGaussian=False,
    #                                      useFlipping=False,useNormData=True, threadLockVar=trainingLock,useCache=True,cache=valid_cache_mem)
    #    validgen.prepareDirectFileInput([validation_directory])

    nsteps = int(math.ceil(epoch_steps / batchSize))
    valid_nsteps = min([int(train_imX_mat.shape[0] / batchSize), int(128 / batchSize)])

    datagen = ScatterPhantomGenerator_inMemory(train_imX_mat,train_imY_mat,batchSize,input_image_size,input_channels,target_image_size,output_channels,
        useResize=True,useCrop=False,useZoom=False,useAWGN=False,useMedian=False,useGaussian=False,
        useFlipping=False,useNormData=True, threadLockVar=trainingLock,save_to_dir="D:\\mbusi\\SCNN\\dumpfold\\train\\")
    datagen.set_nsteps(nsteps)

    validgen = ScatterPhantomGenerator_inMemory(validate_imX_mat,validate_imY_mat,batchSize,input_image_size,input_channels,target_image_size,output_channels,
        useResize=True,useCrop=False,useZoom=False,useAWGN=False,useMedian=False,useGaussian=False,
        useFlipping=False,useNormData=True, threadLockVar=trainingLock,save_to_dir="D:\\mbusi\\SCNN\\dumpfold\\valid\\")
    validgen.set_nsteps(valid_nsteps)

    
    callbacks = []
    callbacks.append(ModelCheckpoint(modelpath=modelfile,
                                    weightspath=weightfile,
                                    period=1,
                                    auto_remove_model=True,
                                    auto_remove_weights=True,
                                    keep_period=keep_period))
    

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
                                                use_multiprocessing=True,
                                                workers=8,
                                                callbacks=callbacks)

    with open(mhistfile.format(epoch=epochs), 'wb') as file:
        pickle.dump(model_fitting_history.history, file)
