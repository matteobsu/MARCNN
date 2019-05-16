#!python3
import pickle
import numpy
from scipy import misc
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model
import tensorflow as tf
import itertools

from argparse import ArgumentParser
import sys
import os
import time
from scipy import io
import h5py
import string



if __name__ == '__main__':
    '''
    Show models and images and error rate 
    
    Variables:
        
    '''
    artifact=0 # show artifacts
    optionParser = ArgumentParser(description="visualisation routine for CT denoise DeepLearning")
    optionParser.add_argument("-m","--modelpath",action="store",dest="modelpath",default="",help="path to model parent folder (i.e. parent of output- and model folder)")
    optionParser.add_argument("-I","--inputPath",action="store",dest="inputPath",help="full path to the input images (validation projections)")
    optionParser.add_argument("-O","--outputPath",action="store",dest="outputPath",help="path to store output images")
    optionParser.add_argument("-G","--groundTruthPath",action="store",dest="groundTruthPath",help="full path to the ground truth images")
    optionParser.add_argument("-M","--modelname",action="store",nargs='*',dest="modelname",help="name of the model file(s); if 1 given, just normal vis; if multiple given, then comparison.")
    optionParser.add_argument("-H","--hist_fname",action="store",dest="hist_fname",help="full path with filename to specific history file")
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
    
    histdir = ""
    if(argDict["hist_fname"]!=None):
        Mhistfile=[]
        Mhistfile.append(argDict["hist_fname"])
        histdir = os.path.dirname(argDict["hist_fname"])

    if(argDict["outputPath"]!=None):
        outPath = argDict["outputPath"]

    #filtered back-projection images
    inputProjectionsFilepath = "../data/images/images_sinogram_f5_inv.pkl"
    if(argDict["inputPath"]!=None):
        inputProjectionsFilepath = argDict["inputPath"]
    print(inputProjectionsFilepath)

    groundtruthScatterMap = "../data/images/images_inv.pkl"
    if(argDict["groundTruthPath"]!=None):
        groundtruthScatterMap = argDict["groundTruthPath"]
    print(groundtruthScatterMap)
    
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
    #plt.show()
    #print (type(Mhist))
    
    lenX = 0
    lenY = 0
    (pname, extName) = os.path.splitext(inputProjectionsFilepath)
    if "h5" in extName:
        print("Using HDF5 interface for input ...")
        file = h5py.File(inputProjectionsFilepath,'r')
        imageS = numpy.array(file['data']['value'])
        file.close()
        lenX = imageS.shape[0]
    else:
        imageS=pickle.load(open(inputProjectionsFilepath, "rb"))[0]

    (pname, extName) = os.path.splitext(groundtruthScatterMap)
    if "h5" in extName:
        print("Using HDF5 interface for groundtruth ...")
        file = h5py.File(groundtruthScatterMap,'r')
        imageI = numpy.array(file['data']['value'])
        file.close()
        lenY = imageS.shape[0]
    else:
        imageI=pickle.load(open(groundtruthScatterMap, "rb"))

    if lenX!=lenY:
        print("Array lengths unequal - exiting.")
        exit()

    #imageS=numpy.reshape(imageS,(24299,96,96,1))
    #imageI=numpy.reshape(imageI,(24299,96,96,1))
    imageS=numpy.reshape(imageS,imageS.shape+(1,))
    imageI=numpy.reshape(imageI,imageS.shape+(1,))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    with sess.as_default():
        model = load_model(modelName[0])
        model.load_weights(weightsName[0])
        
        plot_model(model, to_file='model.png', show_shapes=True)
        
        modelComp = None
        if len(modelName)>1:
            modelComp = load_model(modelName[0])

        imgArr = numpy.zeros((min(lenX,50),3,256,256), dtype=numpy.float32)
        errArr = numpy.zeros((min(lenX,50),256,256), dtype=numpy.float32)
        for imagenr in itertools.islice(itertools.count(), 0, min(lenX,50)):
            # here: in-time prediction - could be stored / precalculated ... ?
            start_predict = time.time()
            #inImage = imageS[imagenr:imagenr+1,:,:,:]
            inImage = imageS[imagenr:imagenr+1,:,:]
            test=model.predict(inImage)
            end_predict = time.time()
            print("model prediction took %f seconds" % (end_predict-start_predict))
            #if artifact==1:
            #    img=imageI[imagenr:imagenr+1].squeeze()+test[0:1].squeeze()
            #else:
            #    img=test.squeeze()
            img=test
            plt.figure(imagenr,figsize=(8, 6), dpi=300)
            # input image #
            plt.subplot(131)
            plt.imshow(inImage.squeeze(), cmap='gray')
            plt.title("Input")
            imgArr[imagenr,0,:,:] = inImage.squeeze()
            # predicted image #
            plt.subplot(132)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title("Prediction")
            imgArr[imagenr,1,:,:] = img.squeeze()
            #print(img.shape)    # should be (1,256,256,1)
            # target image #
            plt.subplot(133)
            #imageI[imagenr:imagenr+1]
            #outimage = imageI[imagenr:imagenr+1,:,:,:]
            outimage = imageI[imagenr:imagenr+1,:,:]
            plt.imshow(outimage.squeeze(), cmap='gray')
            plt.title("Target / Groundtruth")
            imgArr[imagenr,2,:,:] = outimage.squeeze()
            
            #if(imagenr==0):
            #    print(outimage)

            #ref = imageI[imagenr:imagenr+1].squeeze()/255.0
            #error = numpy.sum(numpy.square(outimage - img))
            errArr[imagenr,:,:] = numpy.square(outimage.squeeze() - img.squeeze())
            error = numpy.mean(errArr[imagenr,:,:])
            print("mean squared difference error img %d: %f" % (imagenr, error))
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

