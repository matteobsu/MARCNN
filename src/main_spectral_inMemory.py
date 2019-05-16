
"""
Modified UNet training for scatter estimation 

Created on Jul 26, 2018

@author: christian (Christian Kehl)
@mail: chke@dtu.dk
"""

from __future__ import print_function
#from unetwork import UNetFactory
from UNet2D_Maier2018_spectral import UNet2D_Maier2018_spectral
#from UNet2D_Maier2018 import UNet2D_Maier2018
from ScatteringPhantom import ScatterPhantomGenerator_inMemory

from argparse import ArgumentParser
import os
from scipy import io
import h5py
import pickle
import keras
import numpy
from keras.preprocessing.image import ImageDataGenerator
from callbacks import ModelCheckpoint
from data_helper import DataGenerator
from data_helper import StraightDirectoryGenerator
import itertools
import glob


"""
General parameters for learning
"""
input_indicator = "*X*"
output_indicator = "*Y*"
epochs = 100
keep_period = 1
#image_shape = (256,256,1)       # needs to fit with the actual data dim
image_shape = (256,256,32)       # needs to fit with the actual data dim
#image_shape = (437,437,1)       # needs to fit with the actual data dim
#image_shape = (437,437,32)       # needs to fit with the actual data dim
#targetImageShape = (256,256,1)  # free to choose for resizing
targetImageShape = (256,256,32)  # free to choose for resizing
scaleFactor = min(image_shape[0]/targetImageShape[0], image_shape[1]/targetImageShape[1])
#epoch_steps=16384
epoch_steps=8192
batch_size = 16

def loadHDF5(filename):
    """Loading data, reshaping and normalizing.

    Our training data is precomputed and stored on disk in a pickle file. The
    data is stored as a list of black and white images, i.e. a 3D array of
    integers from 0 to 255. We add the color dimensions, even though the length
    is one, and normalize the values between 0 and 1.
    """
    (pname, extName) = os.path.splitext(filename)
    if "h5" in extName:
        #print("Using HDF5 interface ...")
        file = h5py.File(filename,'r')
        data = numpy.array(file['data']['value'])
        file.close()
        return data
    else:
        pass
    return None

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
    
    td_help = "training data directory"
    option_parser.add_argument("-T", "--train-dir",
                               action="store", dest="train-dir",
                               default="../data/train", help=td_help)
    vd_help = "validation data dir"
    option_parser.add_argument("-V", "--valid-dir",
                               action="store", dest="valid-dir",
                               default="../data/train", help=vd_help)
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
    
    # ======================================================================== #
    # ==== L O A D   P R O J E C T I O N S   A N D   S C A T T E R M A P S === #
    # ====    T R A I N I N G   A N D   V A L I D A T I O N   D A T A     ==== #
    # ======================================================================== #
    import re
    inputFileArray_train = []
    scatterMapArray_train = []
    #observeFileArray_train = []
    inputFileArray_validation = []
    scatterMapArray_validation = []
    #observeFileArray_validation = []
    
    for name in glob.glob(os.path.join(training_directory,'*X*.h5')):
        inputFileArray_train.append(name)
    for name in glob.glob(os.path.join(training_directory,'*Y*.h5')):
        scatterMapArray_train.append(name)
    #for name in glob.glob(os.path.join(training_directory,'*Z*.h5')):
    #    observeFileArray_train.append(name)
    for name in glob.glob(os.path.join(validation_directory,'*X*.h5')):
        inputFileArray_validation.append(name)
    for name in glob.glob(os.path.join(validation_directory,'*Y*.h5')):
        scatterMapArray_validation.append(name)
    #for name in glob.glob(os.path.join(validation_directory,'*Z*.h5')):
    #    observeFileArray_validation.append(name)
        
    digits = re.compile(r'(\d+)')
    def tokenize(filename):
        return tuple(int(token) if match else token for token, match in ((fragment, digits.search(fragment)) for fragment in digits.split(filename)))

    inputFileArray_train.sort(key=tokenize)
    scatterMapArray_train.sort(key=tokenize)
    #observeFileArray_train.sort(key=tokenize)
    inputFileArray_validation.sort(key=tokenize)
    scatterMapArray_validation.sort(key=tokenize)
    #observeFileArray_validation.sort(key=tokenize)
    len_train = len(inputFileArray_train)
    len_validation = len(inputFileArray_validation)
    
    if len_train == 0:
        print("No training data found. Exiting.")
        exit()
    if len_validation == 0:
        print("No validation data found. Exiting.")
        exit()
    dumpDataFile = h5py.File(inputFileArray_train[0], 'r')
    #dumpData = numpy.array(dumpDataFile['dataX'], order='F').transpose()
    dumpData = numpy.array(dumpDataFile['data']['value'], dtype=numpy.float32)
    dumpDataFile.close()
    
    if len(dumpData.shape) < 3:
        input_train = numpy.zeros((len_train, dumpData.shape[0], dumpData.shape[1], 1), dtype=numpy.float32)
        scattermap_train = numpy.zeros((len_train, dumpData.shape[0], dumpData.shape[1], 1), dtype=numpy.float32)
    #    observation_train = numpy.zeros((len_train, dumpData.shape[0], dumpData.shape[1], 1), dtype=numpy.float32)
        for i in itertools.islice(itertools.count(), 0, len_train):
            input_train[i,:,:,0] = loadHDF5(inputFileArray_train[i])
            scattermap_train[i,:,:,0] = loadHDF5(scatterMapArray_train[i])
    #        observation_train[i,:,:,0] = loadHDF5(observeFileArray_train[i])
        input_validation = numpy.zeros((len_validation, dumpData.shape[0], dumpData.shape[1], 1), dtype=numpy.float32)
        scattermap_validation = numpy.zeros((len_validation, dumpData.shape[0], dumpData.shape[1], 1), dtype=numpy.float32)
    #    observation_validation = numpy.zeros((len_validation, dumpData.shape[0], dumpData.shape[1], 1), dtype=numpy.float32)
        for i in itertools.islice(itertools.count(), 0, len_validation):
            input_validation[i,:,:,0] = loadHDF5(inputFileArray_validation[i])
            scattermap_validation[i,:,:,0] = loadHDF5(scatterMapArray_validation[i])
    #        observation_validation[i,:,:,0] = loadHDF5(observeFileArray_validation[i])
    elif len(dumpData.shape) < 4:
        input_train = numpy.zeros((len_train, dumpData.shape[0], dumpData.shape[1], dumpData.shape[2]), dtype=numpy.float32)
        scattermap_train = numpy.zeros((len_train, dumpData.shape[0], dumpData.shape[1], dumpData.shape[2]), dtype=numpy.float32)
    #    observation_train = numpy.zeros((len_train, dumpData.shape[0], dumpData.shape[1], dumpData.shape[2]), dtype=numpy.float32)
        for i in itertools.islice(itertools.count(), 0, len_train):
            input_train[i] = loadHDF5(inputFileArray_train[i])
            scattermap_train[i] = loadHDF5(scatterMapArray_train[i])
    #        observation_train[i] = loadHDF5(observeFileArray_train[i])
        input_validation = numpy.zeros((len_validation, dumpData.shape[0], dumpData.shape[1], dumpData.shape[2]), dtype=numpy.float32)
        scattermap_validation = numpy.zeros((len_validation, dumpData.shape[0], dumpData.shape[1], dumpData.shape[2]), dtype=numpy.float32)
    #    observation_validation = numpy.zeros((len_validation, dumpData.shape[0], dumpData.shape[1], dumpData.shape[2]), dtype=numpy.float32)
        for i in itertools.islice(itertools.count(), 0, len_validation):
            input_validation[i] = loadHDF5(inputFileArray_validation[i])
            scattermap_validation[i] = loadHDF5(scatterMapArray_validation[i])
    #        observation_validation[i] = loadHDF5(observeFileArray_validation[i])
    else:
        input_train = loadHDF5(inputFileArray_train[i])
        scattermap_train = loadHDF5(scatterMapArray_train[i])
    #    observation_train = loadHDF5(observeFileArray_train[i])
        input_validation = loadHDF5(inputFileArray_validation[i])
        scattermap_validation = loadHDF5(scatterMapArray_validation[i])
    #    observation_validation = loadHDF5(observeFileArray_validation[i])
    
    latest_epoch, latest_model = 0, None
    ModelCheckpoint.remove_all_checkpoints(modelfile, weightfile)
    print("Number of data points: %d." % input_train.shape[0])

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
    unet = UNet2D_Maier2018_spectral()
    #unet = UNet2D_Maier2018()
    unet.begin(image_shape=targetImageShape)
    unet.buildNetwork(targetImageShape)
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
    augment = arg_dict["augment"]
    if augment:
        ##datagen = DataGenerator(image_shape, useAWGN=False, 0.05, 0.025, useRotation=False, rotationRange=(0,360), targetSize=(targetImageShape[0],targetImageShape[1]), useZoom=False, zoomFactor=scaleFactor, useFlipping=True)
        #datagen = DataGenerator(image_shape, useAWGN=False, useRotation=False, targetSize=(targetImageShape[0],targetImageShape[1]), useZoom=False, useFlipping=True)
        ##datagen = DataGenerator(image_shape, useAWGN=False, useRotation=False, targetSize=(targetImageShape[0],targetImageShape[1]), useZoom=False, useFlipping=False)
        ##datagen = DataGenerator(image_shape, useAWGN=False, useRotation=True, rotationRange=(0,45), targetSize=(targetImageShape[0],targetImageShape[1]), useZoom=False, useFlipping=True)
        ##datagen.prepareDirectFileInput([training_directory])
        
        #datagen = ScatterPhantomGenerator_inMemory(input_train,observation_train,image_shape,batch_size, useAWGN=False, useRotation=False, targetSize=(targetImageShape[0],targetImageShape[1]), useZoom=False, useFlipping=True)
        datagen = ScatterPhantomGenerator_inMemory(input_train,scattermap_train,image_shape,batch_size, useAWGN=False, useRotation=False, targetSize=(targetImageShape[0],targetImageShape[1]), useZoom=False, useFlipping=True)
    
    callbacks = []
    callbacks.append(ModelCheckpoint(modelpath=modelfile,
                                    weightspath=weightfile,
                                    period=1,
                                    auto_remove_model=True,
                                    auto_remove_weights=True,
                                    keep_period=keep_period))
    if augment:
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

        #model_fitting_history = model.fit_generator(datagen.flow(training_input,
        #                                 training_output,
        #                                 batch_size=16,shuffle=True),
        #                    steps_per_epoch=len(training_input),
        #                    validation_data=(validation_input,
        #                                     validation_output),
        #                    epochs=epochs,
        #                    initial_epoch=latest_epoch,
        #                    use_multiprocessing=True,
        #                    workers=12,
        #                    callbacks=callbacks)
        # datagen.flow(training_input,training_output,len(training_input),16,True)
        
        #model_fitting_history = model.fit_generator(datagen.flow(input_train, scattermap_train, len_train, 20, shuffle=True),
        #                                            steps_per_epoch=int(epoch_steps/batch_size),
        #                                            validation_data=(input_validation, scattermap_validation),
        #                                            epochs=epochs,
        #                                            use_multiprocessing=True,
        #                                            workers=8,
        #                                            callbacks=callbacks)

        #model_fitting_history = model.fit_generator(datagen.flow(input_train, observation_train, len_train, 20, shuffle=True),
        #                                            steps_per_epoch=int(epoch_steps/batch_size),
        #                                            validation_data=(input_validation, observation_validation),
        #                                            epochs=epochs,
        #                                            use_multiprocessing=True,
        #                                            workers=8,
        #                                            callbacks=callbacks)

        model_fitting_history = model.fit_generator(datagen,
                                                    steps_per_epoch=int(epoch_steps/batch_size),
                                                    validation_data=(input_validation, scattermap_validation),
        #                                            validation_steps = int(scattermap_validation.shape[0]/batch_size),
                                                    epochs=epochs,
                                                    use_multiprocessing=True,
                                                    workers=12,
                                                    callbacks=callbacks)
    else:
        model_fitting_history = model.fit(input_train,
                                          scattermap_train,
                                          validation_data=(input_validation, scattermap_validation),
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          initial_epoch=latest_epoch,
                                          callbacks=callbacks)

        #model_fitting_history = model.fit(input_train,
        #                                  observation_train,
        #                                  validation_data=(input_validation, observation_validation),
        #                                  batch_size=batch_size,
        #                                  epochs=epochs,
        #                                  initial_epoch=latest_epoch,
        #                                  callbacks=callbacks)
    
    with open(mhistfile.format(epoch=epochs), 'wb') as file:
        pickle.dump(model_fitting_history.history, file)
