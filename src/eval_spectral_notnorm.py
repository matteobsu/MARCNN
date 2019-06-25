#!python3
import pickle
import numpy
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf
import itertools
from skimage import transform

from argparse import ArgumentParser
import os
import re
import time
import h5py
import glob

TYPES = {"XRAY": 0, "SCATTER": 1, "CT": 2}

def numpy_normalize(v):
    norm = numpy.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

slice_size = (96,96)
input_channels = 32
target_channels = 32
useNormData=False

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

    itype = None
    otype = None
    dumpDataFile = h5py.File(inputFileArray[0], 'r')
    dumpData_in = numpy.array(dumpDataFile['Data_X'], order='F').transpose()
    dumpDataFile.close()
    if len(dumpData_in.shape) > 3:
        dumpData_in = numpy.squeeze(dumpData_in[:, :, 0, :])
    if len(dumpData_in.shape) < 3:
        dumpData_in = dumpData_in.reshape(dumpData_in.shape + (1,))
    # === we need to feed the data as 3D+1 channel data stack === #
    if len(dumpData_in.shape) < 4:
        dumpData_in = dumpData_in.reshape(dumpData_in.shape + (1,))
    channelNum_in = dumpData_in.shape[2]
    print("input shape: {}".format(dumpData_in.shape))
    itype = dumpData_in.dtype
    sqShape_in = numpy.squeeze(dumpData_in).shape
    shape_in = dumpData_in.shape
    shape_process_in = sqShape_in+(1,)

    dumpDataFile = h5py.File(inputFileArray[0], 'r')
    dumpData_out = numpy.array(dumpDataFile['Data_Y'], order='F').transpose()
    dumpDataFile.close()
    if len(dumpData_out.shape) > 3:
        dumpData_out = numpy.squeeze(dumpData_out[:, :, 0, :])
    if len(dumpData_out.shape) < 3:
        dumpData_out = dumpData_out.reshape(dumpData_out.shape + (1,))
    # === we need to feed the data as 3D+1 channel data stack === #
    if len(dumpData_out.shape) < 4:
        dumpData_out = dumpData_out.reshape(dumpData_out.shape + (1,))
    channelNum_out = dumpData_out.shape[2]
    print("scatter shape: {}".format(dumpData_in.shape))
    otype = dumpData_out.dtype
    sqShape_out = numpy.squeeze(dumpData_out).shape
    shape_out = dumpData_out.shape
    shape_process_out = sqShape_out+(1,)

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
        for chidx in range(0,channelNum_in):
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
        
        #plot_model(model, to_file='model.png', show_shapes=True)
        
        modelComp = None
        if len(modelName)>1:
            modelComp = load_model(modelName[0])

        imgArr = numpy.zeros((lenX, 3, slice_size[0], slice_size[1], input_channels), dtype=otype)
        errArr = numpy.zeros((lenX, slice_size[0], slice_size[1], input_channels), dtype=otype)        
        transform_shape_in = (slice_size[0], slice_size[1], shape_process_in[2], shape_process_in[3])
        transform_shape_out = (slice_size[0], slice_size[1], shape_process_out[2], shape_process_out[3])


        #--------------------------------------------------#
        #-- Flat field: reduce energy dimension & resize --#
        #flatFieldSmall = numpy.zeros((96, 96, targetChannels), dtype=otype)
        #for source_Eidx in range(0, 32, energy_bin_range):
        #    target_Eidx = int(source_Eidx / energy_bin_range)
        #    flatFieldSmall[:, :, target_Eidx] = numpy.mean(flatField[:, :, source_Eidx:source_Eidx + energy_bin_range], axis=2)
        #--------------------------------------------------#
        for imagenr in itertools.islice(itertools.count(), 0, lenX):

            file = h5py.File(inputFileArray[imagenr],'r')
            inImage = numpy.array(file['Data_X'], order='F').transpose()
            file.close()
            predictIn = numpy.array(inImage)         
            predictIn = transform.resize(predictIn, (slice_size[0], slice_size[1], channelNum_in), order=3, mode='reflect')
            predictIn = predictIn.reshape(transform_shape_in)              
            predictIn = predictIn.reshape((1,) + predictIn.shape)
            predictIn = predictIn.astype(numpy.float32)

            file = h5py.File(inputFileArray[imagenr], 'r')
            outImage = numpy.array(file['Data_Y'], order='F').transpose()
            file.close()            
            outImage = transform.resize(outImage, (slice_size[0], slice_size[1],outImage.shape[2]), order=3, mode='reflect')
            outImage = outImage.reshape(transform_shape_out)             
            outImage = outImage.reshape((1,) + outImage.shape)
            outImage = outImage.astype(numpy.float32)

            #==========================================================================================================#
            # ====              E N D   N O R M A L I S A T I O N   &   P R E P R O C E S S I N G                 ==== #
            #==========================================================================================================#

            start_predict = time.time()
            img=model.predict(predictIn)
            end_predict = time.time()
            print("model prediction took %f seconds" % (end_predict-start_predict))
            
            print("input shape: {}".format(predictIn.shape))
            
            if fullRun==False:
                plt.figure(imagenr,figsize=(8, 3), dpi=300)
            
            # input image #
            predictIn = predictIn.astype(otype)                
            imgArr[imagenr,0,:,:,:] = predictIn[:,:,:,0]
            
            if fullRun==False:
                plt.subplot(151)
                plt.imshow(predictIn[:,:,0,0].squeeze(), cmap='gray')
                plt.title("Metal affected")
            
            # predicted image #
            img = img.astype(otype)                
            imgArr[imagenr, 1, :, :, :] = img[:,:,:,0]
            
            if fullRun==False:
                plt.subplot(152)
                plt.imshow(img[:,:,0,0].squeeze(), cmap='gray')
                plt.title("MAR prediction")
            
            # target image #                
            outImage = outImage.astype(otype)
            imgArr[imagenr,2,:,:,:] = outImage[:,:,:,0]
            
            if fullRun==False:
                plt.subplot(153)
                plt.imshow(outImage[:,:,0,0].squeeze(), cmap='gray')
                plt.title("Ground Truth")            
            
            errArr[imagenr, :, :, :] = (outImage - img)[:, :, :, 0]
            imgErr = errArr[imagenr,:,:,:]
            normErr = None
            if channelNum_out <=1:
                normErr = numpy_normalize(imgErr).astype(numpy.float32)
            else:
                normErr = numpy.zeros((imgErr.shape[0],imgErr.shape[1],imgErr.shape[2]), dtype=numpy.float32)
                for channelIdx in itertools.islice(itertools.count(), 0, channelNum_out):
                    normErr[:,:,channelIdx] = numpy_normalize(imgErr[:,:,channelIdx]).astype(numpy.float32)
            errArr[imagenr,:,:,:] = numpy.square(errArr[imagenr,:,:,:])
            imgErr = numpy.square(imgErr)
            imgError = numpy.mean(imgErr)
            normErr = numpy.square(normErr)
            normError = numpy.mean(normErr)
            print("MSE img %d: %g" % (imagenr, imgError))
            print("normalized MSE img %d: %g" % (imagenr, normError))
            
            if fullRun:
                for channelIdx in itertools.islice(itertools.count(), 0, channelNum_in):
                    imgName = "predict%04d.png" % imagenr
                    imgPath = os.path.join(outPath,"channel%03d"%(channelIdx),imgName)
                    plt.figure(imagenr,figsize=(8, 3), dpi=300)
                    
                    plt.subplot(151)
                    plt.imshow(predictIn[:,:,channelIdx,0].squeeze(), cmap='gray')
                    plt.title("Metal affected")
                    plt.subplot(152)
                    plt.imshow(img[:,:,channelIdx,0].squeeze(), cmap='gray')
                    plt.title("MAR prediction")
                    plt.subplot(153)
                    plt.imshow(outImage[:,:,channelIdx,0].squeeze(), cmap='gray')
                    plt.title("Ground Truth")
                    
                    plt.savefig(imgPath, dpi=300, format="png")
                    plt.close("all")
            else:
                imgName = "predict%04d.png" % imagenr
                plt.savefig(os.path.join(outPath,imgName), dpi=300, format="png")
            plt.close("all")
            
        imgArrFileName = "images_prediction.h5"
        imgArrFile = h5py.File(os.path.join(outPath,imgArrFileName),'w')
        imgArrFile.create_dataset('data', data=imgArr.transpose());
        imgArrFile.close()
        errArrFileName = "prediction_error.h5"
        errArrFile = h5py.File(os.path.join(outPath,errArrFileName),'w')
        errArrFile.create_dataset('data', data=errArr.transpose());
        errArrFile.close()

