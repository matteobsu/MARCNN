#!python3
import pickle
import numpy
from scipy import misc
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model
import tensorflow as tf
import itertools
from skimage import util
from skimage import transform

from argparse import ArgumentParser
import sys
import os
import re
import time
from scipy import io
import h5py
import string
import glob

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

slice_size = (256,256)
input_channels = 64
target_channels = 64

if __name__ == '__main__':
    '''
    Show models and images and error rate 
    
    Variables:
        
    '''
    fullRun=False
    artifact=0 # show artifacts
    optionParser = ArgumentParser(description="visualisation routine for CT denoise DeepLearning")
    optionParser.add_argument("-m","--modelpath",action="store",dest="modelpath",default="",help="path to model parent folder (i.e. parent of output- and model folder)")
    optionParser.add_argument("-D","--dataPath",action="store",dest="dataPath",help="full path to the evlauation data")
    optionParser.add_argument("-O","--outputPath",action="store",dest="outputPath",help="path to store output images")
    optionParser.add_argument("-M","--modelname",action="store",nargs='*',dest="modelname",help="name of the model file(s); if 1 given, just normal vis; if multiple given, then comparison.")
    optionParser.add_argument("-H","--hist_fname",action="store",dest="hist_fname",help="full path with filename to specific history file")
    optionParser.add_argument("-c","--complete",action="store_true",dest="fullRun",help="enable option to store ALL channels as images")
    #options = optionParser.parse_args(args=sys.argv)
    options = optionParser.parse_args()

    argDict = vars(options)
    
    outPath = ""
    modelName = ["../data/models/unet_imagesinv_strid1_relu_v1_5_64_2_model.h5",] # Model file
    weightsName = ["../data/models/unet_imagesinv_strid1_relu_v1_5_64_2_weights.h5",] # weights file
    Mhistfile = ["../data/output/unet_imagesinv_strid1_relu_v1_5_64_2_Thist.pkl",] # Optimization Error History file
    if("modelpath" in argDict) and (argDict["modelpath"]!=None):
        if("modelname" in argDict) and (argDict["modelname"]!=None):
            modelName=[]
            weightsName=[]
            head = argDict["modelpath"]
            for entry in argDict["modelname"]:
                modelName.append(os.path.join(head, "models", entry+"_model.h5"))
                weightsName.append(os.path.join(head, "models", entry+"_weights.h5"))
            outPath = os.path.join(head, os.path.pardir, "output")
    print(modelName)
    print(weightsName)


    #=============================================================================#
    #=== General setup: what is the input/output datatype, what is SE/MECT ... ===#
    #=============================================================================#
    input_type = TYPES["CT"]
    output_type = TYPES["CT"]

    flatField=None
    if ("fullRun" in argDict) and (argDict["fullRun"]!=None) and (argDict["fullRun"]!=False):
        fullRun=True
    
    histdir = ""
    if(argDict["hist_fname"]!=None):
        Mhistfile=[]
        Mhistfile.append(argDict["hist_fname"])
        histdir = os.path.dirname(argDict["hist_fname"])

    if(argDict["outputPath"]!=None):
        outPath = argDict["outputPath"]

    #==========================================================================#
    #====   S T A R T   O F   C O L L E C T I N G   F I L E   P A T H S    ====#
    #==========================================================================#
    inputFileArray = []
    if argDict["dataPath"]==None:
        exit()
    else:
        for name in glob.glob(os.path.join(argDict["dataPath"],'*.h5')):
            inputFileArray.append(name)

    otype = None
    dumpDataFile = h5py.File(inputFileArray[0], 'r')
    dumpData = numpy.array(dumpDataFile['Data_X'], order='F').transpose()
    dumpDataFile.close()
    slice_size = (dumpData.shape[0], dumpData.shape[1])
    channelNum = numpy.squeeze(dumpData).shape[2]
    otype = dumpData.dtype
    sqShape = numpy.squeeze(dumpData).shape

    # sort the names
    digits = re.compile(r'(\d+)')
    def tokenize(filename):
        return tuple(int(token) if match else token for token, match in ((fragment, digits.search(fragment)) for fragment in digits.split(filename)))
    # Now you can sort your file names like so:
    inputFileArray.sort(key=tokenize)
    #==========================================================================#
    #====     E N D   O F   C O L L E C T I N G   F I L E   P A T H S     =====#
    #==========================================================================#

    
    Mhist=pickle.load(open( Mhistfile[0], "rb" ) )
    # Comment CK: the Mhist file represents (as it seems) the 'metrics' field of a
    # Keras model (see: https://keras.io/models/model/ and https://keras.io/metrics/).
    #print(Mhist.keys())
    # summarize history for error function
    plt.figure("erf",figsize=(6, 6), dpi=300)
    plt.plot(Mhist['mean_squared_error'])
    plt.plot(Mhist['val_mean_squared_error'])
    #plt.plot(Mhist['val_loss'])
    plt.title('mean squared error')
    plt.ylabel('erf(x)')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(outPath,"meanAbsErr.png"), dpi=300, format="png")
    plt.close("all")
    mseArray = numpy.array([Mhist['mean_squared_error'], Mhist['val_mean_squared_error']])
    mseFile = h5py.File(os.path.join(outPath,"mean_squared_error.h5"),'w')
    mseFile.create_dataset('data', data=mseArray.transpose());
    mseFile.close() 
    #plt.show()
    # summarize history for loss
    plt.figure("loss",figsize=(6, 6), dpi=300)
    plt.plot(Mhist['loss'])
    plt.plot(Mhist['val_loss'])
    #plt.plot(Mhist['val_mean_absolute_error'])
    plt.title('loss function')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(outPath,"loss.png"), dpi=300, format="png")
    plt.close("all")
    maeArray = numpy.array([Mhist['loss'], Mhist['val_loss']])
    maeFile = h5py.File(os.path.join(outPath,"loss_mae.h5"),'w')
    maeFile.create_dataset('data', data=maeArray.transpose());
    maeFile.close()
    del mseArray
    del maeArray
    
    lenX = len(inputFileArray)

    if fullRun:
        #outPath
        for chidx in range(0,channelNum):
            chpath = os.path.join(outPath,"channel%03d"%(chidx))
            if os.path.exists(chpath)==False:
                os.mkdir(chpath)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    with sess.as_default():
        model = load_model(modelName[0])
        print(modelName[0])
        model.load_weights(weightsName[0])
        print(weightsName[0])
        
        modelComp = None
        if len(modelName)>1:
            modelComp = load_model(modelName[0])

        #imgArr = numpy.zeros((min(lenX,50),3,256,256), dtype=numpy.float32)
        #errArr = numpy.zeros((min(lenX,50),256,256), dtype=numpy.float32)

        #imgArr = numpy.zeros((min(lenX,50),5,256,256,targetChannels), dtype=otype)
        #errArr = numpy.zeros((min(lenX,50),256,256,targetChannels), dtype=otype)
        #targetChannels = 32
        #energy_bin_range = int(32 / targetChannels)
        #tmp = numpy.zeros((256, 256, targetChannels), dtype=otype)
        imgArr = numpy.zeros((lenX, 5, slice_size[0], slice_size[1], input_channels), dtype=otype)
        errArr = numpy.zeros((lenX, slice_size[0], slice_size[1], input_channels), dtype=otype)


        #--------------------------------------------------#
        #-- Flat field: reduce energy dimension & resize --#
        #flatFieldSmall = numpy.zeros((96, 96, targetChannels), dtype=otype)
        #for source_Eidx in range(0, 32, energy_bin_range):
        #    target_Eidx = int(source_Eidx / energy_bin_range)
        #    flatFieldSmall[:, :, target_Eidx] = numpy.mean(flatField[:, :, source_Eidx:source_Eidx + energy_bin_range], axis=2)
        #--------------------------------------------------#
        for imagenr in itertools.islice(itertools.count(), 0, lenX):
            minValX = None
            maxValX = None
            minValY = None
            maxValY = None

            file = h5py.File(inputFileArray[imagenr],'r')
            # adaptation for reading the non-normalized data
            inImage = numpy.array(file['Data_X'], order='F').transpose()
            predictIn = numpy.array(inImage)
            file.close()
            inImage = transform.resize(inImage, (slice_size[0], slice_size[1],target_channels), order=3, mode='reflect')
            # -- reduce energy dimension & resize -- #
            #for source_Eidx in range(0, 32, energy_bin_range):
            #    target_Eidx = int(source_Eidx / energy_bin_range)
            #    tmp[:, :, target_Eidx] = numpy.mean(inImage[:, :, source_Eidx:source_Eidx + energy_bin_range],axis=2)
            #inImage = numpy.array(tmp)
            minValX, maxValX, inImage = normaliseFieldArray(inImage, target_channels, flatField, input_type)

            #predictIn = transform.resize(numpy.transpose(predictIn), (target_channels, slice_size[0], slice_size[1]), order=3, mode='reflect')
            predictIn = transform.resize(predictIn, (slice_size[0], slice_size[1], target_channels), order=3, mode='reflect')
            PminValX, PmaxValX, predictIn = normaliseFieldArray(predictIn, input_channels, flatField, input_type)
            # end of reduce energy dimension & resize #
            inImage = inImage.astype(numpy.float32)
            inImage = inImage.reshape((1,) + inImage.shape)
            if len(inImage.shape) < 4:
                inImage = inImage.reshape(inImage.shape + (1,))

            predictIn = predictIn.astype(numpy.float32)
            predictIn = predictIn.reshape((1,) + predictIn.shape)
            #predictIn = predictIn.reshape(predictIn.shape + (1,))
            if len(predictIn.shape) < 4:
            #    predictIn = predictIn.reshape((1,) + predictIn.shape)
                predictIn = predictIn.reshape(predictIn.shape + (1,))

            file = h5py.File(inputFileArray[imagenr], 'r')
            # adaptation for reading the non-normalized data
            outImage = numpy.array(file['Data_Y'], order='F').transpose()
            file.close()
            outImage = transform.resize(outImage, (slice_size[0], slice_size[1],target_channels), order=3, mode='reflect')
            # reduce energy dimension & resize #
            #for source_Eidx in range(0, 32, energy_bin_range):
            #    target_Eidx = int(source_Eidx / energy_bin_range)
            #    tmp[:, :, target_Eidx] = numpy.mean(outImage[:, :, source_Eidx:source_Eidx + energy_bin_range],axis=2)
            #outImage = numpy.array(tmp)
            minValY, maxValY, outImage = normaliseFieldArray(outImage, target_channels, flatField, output_type)

            # end of reduce energy dimension & resize #
            outImage = outImage.astype(numpy.float32)
            outImage = outImage.reshape((1,) + outImage.shape)
            if len(outImage.shape) < 4:
                outImage = outImage.reshape(outImage.shape + (1,))


            #==========================================================================================================#
            # ====              E N D   N O R M A L I S A T I O N   &   P R E P R O C E S S I N G                 ==== #
            #==========================================================================================================#

            start_predict = time.time()
            predictOut = numpy.zeros(predictIn.shape, dtype=predictIn.dtype)
            predictIn_shape = (1,predictIn.shape[1],predictIn.shape[2],1)
            for channelIdx in itertools.islice(itertools.count(), 0, target_channels):
                prediction_slice = numpy.reshape(predictIn[:,:,:,channelIdx], predictIn_shape)
                #print(prediction_slice.shape)
                # here: in-time prediction - could be stored / precalculated ... ?
                img=model.predict(prediction_slice)
                # predict scatter map instead of the corrected image
                # img = inImage - img
                #print(img.shape)
                predictOut[0,:,:,channelIdx] = img[0,:,:,0]
            end_predict = time.time()
            print("model prediction took %f seconds" % (end_predict-start_predict))
            #figs = []
            if fullRun==False:
                plt.figure(imagenr,figsize=(8, 5), dpi=300)
            
            # input image #
            if fullRun==False:
                plt.subplot(131)
                plt.imshow(inImage[0,:,:,0].squeeze(), cmap='gray')
                plt.title("metal-affected")
            #imgArr[imagenr,0,:,:,:] = inImage.squeeze()
            inImage = inImage.astype(otype).squeeze()
            # == Denormalization == #
            #if NORM_DATA_MODE == 0:
            #    inImage = inImage*(maxValX-minValX)+minValX
            #elif NORM_DATA_MODE == 1:
            #    for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
            #        inImage[:,:,channelIdx] = inImage[:,:,channelIdx]*(maxValX[channelIdx]-minValX[channelIdx])+minValX[channelIdx]
            #elif NORM_DATA_MODE == 2:
            #    inImage = inImage*flatField
            inImage = denormaliseFieldArray(inImage, target_channels, minValX, maxValX, flatField, input_type)
            imgArr[imagenr,0,:,:,:] = inImage
            
            # predicted image #
            if fullRun==False:
                plt.subplot(132)
                plt.imshow(predictOut[0,:,:,0].squeeze(), cmap='gray')
                plt.title("MAR predicted")
            #imgArr[imagenr,1,:,:,:] = img.squeeze()
            predictOut = predictOut.astype(otype).squeeze()
            # == Denormalization == #
            #if NORM_DATA_MODE == 0:
            #    img = img*(maxValY-minValY)+minValY
            #elif NORM_DATA_MODE == 1:
            #    for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
            #        img[:,:,channelIdx] = img[:,:,channelIdx]*(maxValY[channelIdx]-minValY[channelIdx])+minValY[channelIdx]
            #elif NORM_DATA_MODE == 2:
            #    for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
            #        img[:,:,channelIdx] = img[:,:,channelIdx]*(maxValY[channelIdx]-minValY[channelIdx])+minValY[channelIdx]
            predictOut = denormaliseFieldArray(predictOut, target_channels, minValY, maxValY, flatField, output_type)
            #img = transform.resize(img[:,:,:], (256, 256, predictIn.shape[3]), order=3, mode='reflect')
            imgArr[imagenr,1,:,:,:] = predictOut

            # target image #
            if fullRun==False:
                plt.subplot(133)
                plt.imshow(outImage[0,:,:,0].squeeze(), cmap='gray')
                plt.title("MAR groundtruth")
            #imgArr[imagenr,4,:,:,:] = outImage.squeeze()
            outImage = outImage.astype(otype).squeeze()
            # == Denormalization == #
            #if NORM_DATA_MODE == 0:
            #    outImage = outImage*(maxValY-minValY)+minValY
            #elif NORM_DATA_MODE == 1:
            #    for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
            #        outImage[:,:,channelIdx] = outImage[:,:,channelIdx]*(maxValY[channelIdx]-minValY[channelIdx])+minValY[channelIdx]
            #elif NORM_DATA_MODE == 2:
            #    outImage = outImage*flatField
            outImage = denormaliseFieldArray(outImage, target_channels, minValY, maxValY, flatField, output_type)
            imgArr[imagenr,2,:,:,:] = outImage

            #errArr[imagenr,:,:] = numpy.square(outImage.squeeze() - img.squeeze())
            #errArr[imagenr,:,:,:] = numpy.square(outImage.squeeze() - img.squeeze())
            errArr[imagenr,:,:,:] = outImage - predictOut
            imgErr = errArr[imagenr,:,:,:]
            normErr = None
            if target_channels <=1:
                normErr = numpy_normalize(imgErr).astype(numpy.float32)
            else:
                normErr = numpy.zeros((imgErr.shape[0],imgErr.shape[1],imgErr.shape[2]), dtype=numpy.float32)
                for channelIdx in itertools.islice(itertools.count(), 0, target_channels):
                    normErr[:,:,channelIdx] = numpy_normalize(imgErr[:,:,channelIdx]).astype(numpy.float32)
            errArr[imagenr,:,:,:] = numpy.square(errArr[imagenr,:,:,:])
            imgErr = numpy.square(imgErr)
            imgError = numpy.mean(imgErr)
            normErr = numpy.square(normErr)
            normError = numpy.mean(normErr)
            print("MSE img %d: %g" % (imagenr, imgError))
            print("normalized MSE img %d: %g" % (imagenr, normError))

            if fullRun:
                for channelIdx in itertools.islice(itertools.count(), 0, target_channels):
                    imgName = "predict%04d.png" % imagenr
                    imgPath = os.path.join(outPath,"channel%03d"%(channelIdx),imgName)
                    plt.figure(imagenr,figsize=(8, 5), dpi=300)
                    
                    plt.subplot(131)
                    #figs[channelIdx].add_subplot(151)
                    plt.imshow(inImage[:,:,channelIdx].squeeze(), cmap='gray')
                    #figs[channelIdx].figimage(inImage[0,:,:,channelIdx].squeeze(), cmap='gray')
                    plt.title("metal-affected")
                    #figs[channelIdx].suptitle("Scattered In.")
                    plt.subplot(132)
                    #figs[channelIdx].add_subplot(152)
                    plt.imshow(predictOut[:,:,channelIdx].squeeze(), cmap='gray')
                    #figs[channelIdx].figimage(img[0,:,:,channelIdx].squeeze(), cmap='gray')
                    plt.title("MAR predicted")
                    #figs[channelIdx].suptitle("Scatter Pred.")
                    plt.subplot(133)
                    #figs[channelIdx].add_subplot(153)
                    plt.imshow(outImage[:,:,channelIdx].squeeze(), cmap='gray')
                    #figs[channelIdx].figimage(scatterImage[0,:,:,channelIdx].squeeze(), cmap='gray')
                    plt.title("MAR groundtruth")
                    #figs[channelIdx].suptitle("Scatter GT")
                    
                    plt.savefig(imgPath, dpi=300, format="png")
                    plt.close("all")
                    #figs[channelIdx].savefig(imgPath, dpi=300, format="png")
                    #figs[channelIdx].close()
            else:
                imgName = "predict%04d.png" % imagenr
                plt.savefig(os.path.join(outPath,imgName), dpi=300, format="png")
            plt.close("all")
            
        imgArrFileName = "images_prediction.h5"
        imgArrFile = h5py.File(os.path.join(outPath,imgArrFileName),'w')
        imgArrFile.create_dataset('data', data=imgArr.transpose())
        imgArrFile.close()
        errArrFileName = "prediction_error.h5"
        errArrFile = h5py.File(os.path.join(outPath,errArrFileName),'w')
        errArrFile.create_dataset('data', data=errArr.transpose())
        errArrFile.close()

