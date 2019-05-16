
"""
Modified UNet training for scatter estimation 

Created on Jul 06, 2018

@author: christian (Christian Kehl)
@mail: chke@dtu.dk
"""

from __future__ import print_function
#from unetwork import UNetFactory
from UNet2D_Maier2018 import UNet2D_Maier2018
#from UNet2D_Maier2018_spectral import UNet2D_Maier2018_spectral

from argparse import ArgumentParser
import os
from scipy import io
import h5py
import pickle
import keras
import numpy
from keras.preprocessing.image import ImageDataGenerator
from callbacks import ModelCheckpoint
from ScatteringPhantom import ScatterPhantomGenerator
import multiprocessing

"""
General parameters for learning
"""
input_indicator = "*X*"
output_indicator = "*Y*"
epochs = 150
keep_period = 10
input_image_size = (256,256)       # needs to fit with the actual data dim
target_image_size = (256,256)  # free to choose for resizing
input_channels = 64
output_channels = 64
epoch_steps=4096
#batchSize=128
batchSize = input_channels
cache_reuse_period=192
#cache_reuse_period=384
#train_size = 4096
validation_size = 16

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
    vdi_help = "validation input data directory"
    option_parser.add_argument("-V", "--valid-dir-in",
                               action="store", dest="valid-dir-in",
                               default="../data/validation", help=vdi_help)

    ff_help = "flat field file path"
    option_parser.add_argument("-F", "--flatfield-path",
                               action="store", dest="flatfield-path",
                               default=None, help=ff_help)
    
    #c_help = ("name of the continuing, previous model")
    #option_parser.add_argument("-c", "--continue_model", action="store", dest="continue_mfile", default="", help=c_help)
    
    options = option_parser.parse_args()
    arg_dict = vars(options)
    
    training_directory_X = os.path.join(arg_dict["train-dir-in"])
    validation_directory_X = os.path.join(arg_dict["valid-dir-in"])
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
    unet = UNet2D_Maier2018()
    unet.begin((target_image_size[0],target_image_size[1],1))
    unet.buildNetwork()
    model = unet.finalize()

    """
    Adding an optimizer (optional)---can be used to optimize gradient decent.
    We employ the rmsprop: divide the gradient by a running average of its
    recent magnitude.
    These are, for the moment, arbitrarily chosen by the original author. The
    work would benifit from a thorough walk through these choices.
    """
    #opt = keras.optimizers.rmsprop(lr=0.0001,  # Learning rate (lr)
    #                               decay=1e-6)  # lr decay over each update.
    opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6, amsgrad=False)
    model.compile(loss='mean_absolute_error',  # standard cost function
                  optimizer=opt,
                  metrics=['mse'])  # Mean Absolute Error metrics

    """
    Data augmentation setup
    """    
    trainingLock = multiprocessing.Lock()
    validationLock = multiprocessing.Lock()

    # gauss_mu=0.05, gauss_stdDev=0.025,
    datagen = ScatterPhantomGenerator(batchSize,input_image_size,input_channels,target_image_size,output_channels,
                                          useResize=False,useCrop=False,useZoom=False,useAWGN=False,useMedian=False,useGaussian=False,
                                          useFlipping=False,useNormData=True, threadLockVar=trainingLock, useCache=True, cache_period=cache_reuse_period)
    datagen.prepareDirectFileInput(input_image_paths=[training_directory_X])
    validgen = ScatterPhantomGenerator(batchSize,input_image_size,input_channels,target_image_size,output_channels,
                                          useResize=False,useCrop=False,useZoom=False,useAWGN=False,useMedian=False,useGaussian=False,
                                          useFlipping=False,useNormData=True, threadLockVar=trainingLock, useCache=True, cache_period=cache_reuse_period)
    validgen.prepareDirectFileInput(input_image_paths=[validation_directory_X])
    
    callbacks = []
    callbacks.append(ModelCheckpoint(modelpath=modelfile,
                                    weightspath=weightfile,
                                    period=1,
                                    auto_remove_model=True,
                                    auto_remove_weights=True,
                                    keep_period=keep_period))

    model_fitting_history = model.fit_generator(datagen,
                                                    steps_per_epoch=int(epoch_steps/batchSize),
                                                    validation_data=validgen,
                                                    validation_steps=min([int(validgen.numImages/batchSize), int(128/batchSize)]),
                                                    epochs=epochs,
                                                    use_multiprocessing=False,
                                                    #workers=12,
                                                    callbacks=callbacks)
    
    with open(mhistfile.format(epoch=epochs), 'wb') as file:
        pickle.dump(model_fitting_history.history, file)
