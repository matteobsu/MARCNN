#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 2018

@author: chke (Christian Kehl)
"""
from argparse import ArgumentParser
import os
import re
import numpy
import sys
import itertools
import time
import math
from scipy import io
import h5py
import glob
from scipy import ndimage

TRAIN_RATIO = 0.85
VALID_RATIO = 0.1
TEST_RATIO = 0.05
#SCALE_FACTOR = 1.70703125
SCALE_FACTOR = 1.706666667

if __name__ == '__main__':
    optionParser = ArgumentParser(description="tool for fusing separate input files into one concise output field")
    optionParser.add_argument("-i","--inputDir",action="store",dest="inputDir",default="",help="directory with input data files")
    optionParser.add_argument("-o","--outputDir",action="store",dest="outputDir",default="",help="directory for output files")
    optionParser.add_argument("-n","--name",action="store",dest="datasetName",default="",help="name of the experiment")
    optionParser.add_argument("-I","--InputDirs",action="store",nargs='*',dest="inputDirs",help="multiple directories with input data files")
    optionParser.add_argument("-O","--OutputDirs",action="store",nargs='*',dest="outputDirs",help="multiple directories for each train, test and validation data")
    options = optionParser.parse_args()

    argDict = vars(options)
    inputFileArray = []
    scatterFileArray = []
    observationFileArray = []
    inputFileArrayTemp = []
    scatterFileArrayTemp = []
    observationFileArrayTemp = []
    trainOutputDir = ""
    validationOutputDir = ""
    testOutputDir = ""
    
    if("inputDirs" in argDict) and (argDict["inputDirs"]!=None):
        dirArray = []
        for entry in argDict["inputDirs"]:
            dirArray.append(entry)
        for entry in dirArray:
            for name in glob.glob(os.path.join(entry,'*X*.h5')):
                inputFileArrayTemp.append(name)
            for name in glob.glob(os.path.join(entry,'*Y*.h5')):
                scatterFileArrayTemp.append(name)
            for name in glob.glob(os.path.join(entry,'*Z*.h5')):
                observationFileArrayTemp.append(name)
    else:
        for name in glob.glob(os.path.join(options.inputDir,'*X*.h5')):
            inputFileArrayTemp.append(name)
        for name in glob.glob(os.path.join(options.inputDir,'*Y*.h5')):
            scatterFileArrayTemp.append(name)
        for name in glob.glob(os.path.join(options.inputDir,'*Z*.h5')):
            observationFileArrayTemp.append(name)
            
    if("outputDirs" in argDict) and(argDict["outputDirs"]!=None):
        dirArray = []
        for entry in argDict["outputDirs"]:
            dirArray.append(entry)
        trainOutputDir = dirArray[0]
        validationOutputDir = dirArray[1]
        testOutputDir = dirArray[2]
    else:
        trainOutputDir = options.outputDir
        validationOutputDir = options.outputDir
        testOutputDir = options.outputDir

    print("Training data dir: "+trainOutputDir)
    print("Validation data dir: "+validationOutputDir)
    print("Test data dir: "+testOutputDir)

    dumpDataFile = h5py.File(inputFileArrayTemp[0], 'r')
    dumpData = numpy.array(dumpDataFile['dataX'], order='F').transpose()
    dumpDataFile.close()

    #==========================================================================#
    #===   S T A R T   O F   M I S S I N G   F I L E   T R E A T M E N T    ===#
    #==========================================================================#
    # sort the names
    digits = re.compile(r'(\d+)')
    def tokenize(filename):
        return tuple(int(token) if match else token for token, match in ((fragment, digits.search(fragment)) for fragment in digits.split(filename)))
    # Now you can sort your file names like so:
    inputFileArrayTemp.sort(key=tokenize)
    scatterFileArrayTemp.sort(key=tokenize)
    observationFileArrayTemp.sort(key=tokenize)
    
    lenX = len(inputFileArrayTemp)
    lenY = len(scatterFileArrayTemp)
    lenZ = len(observationFileArrayTemp)
    if lenX != lenY or lenX != lenZ or lenY != lenZ:
        print("Number of arrays do not match - input: %d, scatterMap: %d, observation: %d" % (lenX,lenY,lenZ))
        exit()
    numSamples = 0
    missingSamples = 0
    for i in itertools.islice(itertools.count(), 0, lenX):
        inName = inputFileArrayTemp[i]
        scatterName = scatterFileArrayTemp[i]
        outName = observationFileArrayTemp[i]
        #print(name)
        bname = os.path.basename(inName)
        fname = bname.split(".")[0]
        numIn = int(fname.split("_")[1])
        bname = os.path.basename(scatterName)
        fname = bname.split(".")[0]
        numScatter = int(fname.split("_")[1])
        bname = os.path.basename(outName)
        fname = bname.split(".")[0]
        numOut = int(fname.split("_")[1])
        if numIn == numOut == numScatter:
            inputFileArray.append(inName)
            scatterFileArray.append(scatterName)
            observationFileArray.append(outName)
            numSamples+=1

    del inputFileArrayTemp[:]
    del scatterFileArrayTemp[:]
    del observationFileArrayTemp[:]
    print("Given: %d samples, with with %d missing files." % (numSamples,missingSamples))
    #==========================================================================#
    #===     E N D   O F   M I S S I N G   F I L E   T R E A T M E N T     ====#
    #==========================================================================#


    lenX = len(inputFileArray)
    lenY = len(scatterFileArray)
    lenZ = len(observationFileArray)
    if lenX != lenY or lenX != lenZ or lenY != lenZ:
        print("Number of arrays do not match - input: %d, scatterMap: %d, observation: %d" % (lenX,lenY,lenZ))
        
    inputRange = numpy.array([100000.0,-100000.0],dtype=numpy.float64)
    scatterRange = numpy.array([100000.0,-100000.0],dtype=numpy.float64)
    observationRange = numpy.array([100000.0,-100000.0],dtype=numpy.float64)
    TRAIN_LENGTH = int(lenX*TRAIN_RATIO)
    VALIDATION_LENGTH = int(lenX*VALID_RATIO)
    TEST_LENGTH = int(lenX*TEST_RATIO)
    print("%d train data, %d validation data, %d test data." % (TRAIN_LENGTH,VALIDATION_LENGTH,TEST_LENGTH))
    #inputData = numpy.zeros((TRAIN_LENGTH,int(dumpData.shape[0]*1.706666667),int(dumpData.shape[1]*1.706666667)))
    #observationData = numpy.zeros((TRAIN_LENGTH,int(dumpData.shape[0]*1.706666667),int(dumpData.shape[1]*1.706666667)))
    #inputData_validation = numpy.zeros((VALIDATION_LENGTH,int(dumpData.shape[0]*1.706666667),int(dumpData.shape[1]*1.706666667)))
    #observationData_validation = numpy.zeros((VALIDATION_LENGTH,int(dumpData.shape[0]*1.706666667),int(dumpData.shape[1]*1.706666667)))
    #inputData_test = numpy.zeros((TEST_LENGTH,int(dumpData.shape[0]*1.706666667),int(dumpData.shape[1]*1.706666667)))
    #observationData_test = numpy.zeros((TEST_LENGTH,int(dumpData.shape[0]*1.706666667),int(dumpData.shape[1]*1.706666667)))

    endTrain = int(lenX*TRAIN_RATIO)
    endValidation = int(lenX*(TRAIN_RATIO+VALID_RATIO))
    global_data_index = 0
    prtsz = True
    print("=== data - training ===")
    for i in itertools.islice(itertools.count(), 0, endTrain):
        inName = inputFileArray[i]
        scatterName = scatterFileArray[i]
        observeName = observationFileArray[i]
    
        f = h5py.File(inName, 'r')
        img = numpy.array(f['dataX'], order='F').transpose()
        f.close()
        #img = numpy.squeeze(numpy.sum(img, 2))
        #== Note: to be done on data load to reduce load-from-disk time ==#
        #img = ndimage.zoom(img, [SCALE_FACTOR,SCALE_FACTOR,1], order=2)           # for new data with margin: zoom factor = 1.70703125 (256 -> 437); then clip to 256 in preproc.
        #== Note: normalisation - temporarily exempted ==#
        #minVal = numpy.min(img)
        #maxVal = numpy.max(img)
        #img = (img-minVal)/(maxVal-minVal)
        #== Note: for over-sized dataset: crop here ==#
        if img.shape[0] > 150:
            if prtsz == True:
                print(img.shape)
            im_center = numpy.array([(img.shape[0]-1)/2, (img.shape[1]-1)/2], dtype=numpy.int32)
            left = im_center[0]-74
            right = left+150
            top = im_center[1]-74
            bottom = top+150
            img = img[left:right,top:bottom,:]
            if prtsz == True:
                print(img.shape)
                prtsz = False
        if i<TRAIN_LENGTH:
            fname = "%s_%05d_X" % (options.datasetName,global_data_index)
            fOut = h5py.File(os.path.join(trainOutputDir,fname+".h5"),'w')
            outGroup = fOut.create_group('data')
            #outGroup.create_dataset('value', data=img.astype(numpy.float32));
            outGroup.create_dataset('value', data=img);
            fOut.close()

            #== Note: normalisation - temporarily exempted ==#
            #rangeOut = open(os.path.join(trainOutputDir,fname+"_norm"+".txt"),"w")
            #rangeOut.write("min_global: %g\n" % minVal)
            #rangeOut.write("max_global: %g" % maxVal)
            #rangeOut.close()
        del img

        f = h5py.File(scatterName, 'r')
        img = numpy.array(f['dataY'], order='F').transpose()
        f.close()
        #img = numpy.squeeze(numpy.sum(img, 2))
        #== Note: to be done on data load to reduce load-from-disk time ==#
        #img = ndimage.zoom(img, [SCALE_FACTOR,SCALE_FACTOR,1], order=2)           # for new data with margin: zoom factor = 1.70703125 (256 -> 437); then clip to 256 in preproc.
        #== Note: normalisation - temporarily exempted ==#
        #minVal = numpy.min(img)
        #maxVal = numpy.max(img)
        #img = (img-minVal)/(maxVal-minVal)
        #== Note: for over-sized dataset: crop here ==#
        if img.shape[0] > 150:
            if prtsz == True:
                print(img.shape)
            im_center = numpy.array([(img.shape[0]-1)/2, (img.shape[1]-1)/2], dtype=numpy.int32)
            left = im_center[0]-74
            right = left+150
            top = im_center[1]-74
            bottom = top+150
            img = img[left:right,top:bottom,:]
            if prtsz == True:
                print(img.shape)
                prtsz = False
        if i<TRAIN_LENGTH:
            fname = "%s_%05d_Y" % (options.datasetName,global_data_index)
            fOut = h5py.File(os.path.join(trainOutputDir,fname+".h5"),'w')
            outGroup = fOut.create_group('data')
            #outGroup.create_dataset('value', data=img.astype(numpy.float32));
            outGroup.create_dataset('value', data=img);
            fOut.close()

            #== Note: normalisation - temporarily exempted ==#
            #rangeOut = open(os.path.join(trainOutputDir,fname+"_norm"+".txt"),"w")
            #rangeOut.write("min_global: %g\n" % minVal)
            #rangeOut.write("max_global: %g" % maxVal)
            #rangeOut.close()
        del img

        f = h5py.File(observeName, 'r')
        img = numpy.array(f['dataZ'], order='F').transpose()
        f.close()
        #img = numpy.squeeze(numpy.sum(img, 2))
        #== Note: to be done on data load to reduce load-from-disk time ==#
        #img = ndimage.zoom(img, [SCALE_FACTOR,SCALE_FACTOR,1], order=2)
        #== Note: normalisation - temporarily exempted ==#
        #minVal = numpy.min(img)
        #maxVal = numpy.max(img)
        #img = (img-minVal)/(maxVal-minVal)
        #== Note: for over-sized dataset: crop here ==#
        if img.shape[0] > 150:
            if prtsz == True:
                print(img.shape)
            im_center = numpy.array([(img.shape[0]-1)/2, (img.shape[1]-1)/2], dtype=numpy.int32)
            left = im_center[0]-74
            right = left+150
            top = im_center[1]-74
            bottom = top+150
            img = img[left:right,top:bottom,:]
            if prtsz == True:
                print(img.shape)
                prtsz = False
        if i<TRAIN_LENGTH:
            fname = "%s_%05d_Z" % (options.datasetName,global_data_index)
            fOut = h5py.File(os.path.join(trainOutputDir,fname+".h5"),'w')
            outGroup = fOut.create_group('data')
            #outGroup.create_dataset('value', data=img.astype(numpy.float32));
            outGroup.create_dataset('value', data=img);
            fOut.close()

            #== Note: normalisation - temporarily exempted ==#
            #rangeOut = open(os.path.join(trainOutputDir,fname+"_norm"+".txt"),"w")
            #rangeOut.write("min_global: %g\n" % minVal)
            #rangeOut.write("max_global: %g" % maxVal)
            #rangeOut.close()
        del img
            
        if i<TRAIN_LENGTH:
            #fname = "%s_%05d" % (options.datasetName,global_data_index)
            #print("processed entry '%s'" % (fname))
            if ((global_data_index%100) == 0) and (global_data_index!=0):
                print("progress: %02f %%" % (global_data_index*100/lenX))
            global_data_index+=1

    print("=== data - validation ===")
    j=0
    for i in itertools.islice(itertools.count(), endTrain, endValidation):
        inName = inputFileArray[i]
        scatterName = scatterFileArray[i]
        observeName = observationFileArray[i]

        f = h5py.File(inName, 'r')
        img = numpy.array(f['dataX'], order='F').transpose()
        f.close()
        #img = numpy.squeeze(numpy.sum(img, 2))
        #== Note: to be done on data load to reduce load-from-disk time ==#
        #img = ndimage.zoom(img, [SCALE_FACTOR,SCALE_FACTOR,1], order=2)
        #== Note: normalisation - temporarily exempted ==#
        #minVal = numpy.min(img)
        #maxVal = numpy.max(img)
        #img = (img-minVal)/(maxVal-minVal)
        #== Note: for over-sized dataset: crop here ==#
        if img.shape[0] > 150:
            if prtsz == True:
                print(img.shape)
            im_center = numpy.array([(img.shape[0]-1)/2, (img.shape[1]-1)/2], dtype=numpy.int32)
            left = im_center[0]-74
            right = left+150
            top = im_center[1]-74
            bottom = top+150
            img = img[left:right,top:bottom,:]
            if prtsz == True:
                print(img.shape)
                prtsz = False
        if j<VALIDATION_LENGTH:
            fname = "%s_%05d_X" % (options.datasetName,global_data_index)
            fOut = h5py.File(os.path.join(validationOutputDir,fname+".h5"),'w')
            outGroup = fOut.create_group('data')
            #outGroup.create_dataset('value', data=img.astype(numpy.float32));
            outGroup.create_dataset('value', data=img);
            fOut.close()

            #== Note: normalisation - temporarily exempted ==#
            #rangeOut = open(os.path.join(trainOutputDir,fname+"_norm"+".txt"),"w")
            #rangeOut.write("min_global: %g\n" % minVal)
            #rangeOut.write("max_global: %g" % maxVal)
            #rangeOut.close()
        del img

        f = h5py.File(scatterName, 'r')
        img = numpy.array(f['dataY'], order='F').transpose()
        f.close()
        #img = numpy.squeeze(numpy.sum(img, 2))
        #== Note: to be done on data load to reduce load-from-disk time ==#
        #img = ndimage.zoom(img, [SCALE_FACTOR,SCALE_FACTOR,1], order=2)
        #== Note: normalisation - temporarily exempted ==#
        #minVal = numpy.min(img)
        #maxVal = numpy.max(img)
        #img = (img-minVal)/(maxVal-minVal)
        #== Note: for over-sized dataset: crop here ==#
        if img.shape[0] > 150:
            if prtsz == True:
                print(img.shape)
            im_center = numpy.array([(img.shape[0]-1)/2, (img.shape[1]-1)/2], dtype=numpy.int32)
            left = im_center[0]-74
            right = left+150
            top = im_center[1]-74
            bottom = top+150
            img = img[left:right,top:bottom,:]
            if prtsz == True:
                print(img.shape)
                prtsz = False
        if j<VALIDATION_LENGTH:
            fname = "%s_%05d_Y" % (options.datasetName,global_data_index)
            fOut = h5py.File(os.path.join(validationOutputDir,fname+".h5"),'w')
            outGroup = fOut.create_group('data')
            #outGroup.create_dataset('value', data=img.astype(numpy.float32));
            outGroup.create_dataset('value', data=img);
            fOut.close()

            #== Note: normalisation - temporarily exempted ==#
            #rangeOut = open(os.path.join(trainOutputDir,fname+"_norm"+".txt"),"w")
            #rangeOut.write("min_global: %g\n" % minVal)
            #rangeOut.write("max_global: %g" % maxVal)
            #rangeOut.close()
        del img

        f = h5py.File(observeName, 'r')
        img = numpy.array(f['dataZ'], order='F').transpose()
        f.close()
        #img = numpy.squeeze(numpy.sum(img, 2))
        #== Note: to be done on data load to reduce load-from-disk time ==#
        #img = ndimage.zoom(img, [SCALE_FACTOR,SCALE_FACTOR,1], order=2)
        #== Note: normalisation - temporarily exempted ==#
        #minVal = numpy.min(img)
        #maxVal = numpy.max(img)
        #img = (img-minVal)/(maxVal-minVal)
        #== Note: for over-sized dataset: crop here ==#
        if img.shape[0] > 150:
            if prtsz == True:
                print(img.shape)
            im_center = numpy.array([(img.shape[0]-1)/2, (img.shape[1]-1)/2], dtype=numpy.int32)
            left = im_center[0]-74
            right = left+150
            top = im_center[1]-74
            bottom = top+150
            img = img[left:right,top:bottom,:]
            if prtsz == True:
                print(img.shape)
                prtsz = False
        if j<VALIDATION_LENGTH:
            fname = "%s_%05d_Z" % (options.datasetName,global_data_index)
            fOut = h5py.File(os.path.join(validationOutputDir,fname+".h5"),'w')
            outGroup = fOut.create_group('data')
            #outGroup.create_dataset('value', data=img.astype(numpy.float32));
            outGroup.create_dataset('value', data=img);
            fOut.close()

            #== Note: normalisation - temporarily exempted ==#
            #rangeOut = open(os.path.join(trainOutputDir,fname+"_norm"+".txt"),"w")
            #rangeOut.write("min_global: %g\n" % minVal)
            #rangeOut.write("max_global: %g" % maxVal)
            #rangeOut.close()
        del img

        if j<VALIDATION_LENGTH:
            #fname = "%s_%05d" % (options.datasetName,global_data_index)
            #print("processed entry '%s'" % (fname))
            if ((global_data_index%100) == 0) and (global_data_index!=0):
                print("progress: %02f %%" % (global_data_index*100/lenX))
            global_data_index+=1
        j+=1

    print("=== data - testing ===")
    j=0
    for i in itertools.islice(itertools.count(), endValidation, lenX):
        inName = inputFileArray[i]
        scatterName = scatterFileArray[i]
        observeName = observationFileArray[i]

        f = h5py.File(inName, 'r')
        img = numpy.array(f['dataX'], order='F').transpose()
        f.close()
        #img = numpy.squeeze(numpy.sum(img, 2))
        #== Note: to be done on data load to reduce load-from-disk time ==#
        #img = ndimage.zoom(img, [SCALE_FACTOR,SCALE_FACTOR,1], order=2)
        #== Note: normalisation - temporarily exempted ==#
        #minVal = numpy.min(img)
        #maxVal = numpy.max(img)
        #img = (img-minVal)/(maxVal-minVal)
        #== Note: for over-sized dataset: crop here ==#
        if img.shape[0] > 150:
            if prtsz == True:
                print(img.shape)
            im_center = numpy.array([(img.shape[0]-1)/2, (img.shape[1]-1)/2], dtype=numpy.int32)
            left = im_center[0]-74
            right = left+150
            top = im_center[1]-74
            bottom = top+150
            img = img[left:right,top:bottom,:]
            if prtsz == True:
                print(img.shape)
                prtsz = False
        if j<TEST_LENGTH:
            fname = "%s_%05d_X" % (options.datasetName,global_data_index)
            fOut = h5py.File(os.path.join(testOutputDir,fname+".h5"),'w')
            outGroup = fOut.create_group('data')
            #outGroup.create_dataset('value', data=img.astype(numpy.float32));
            outGroup.create_dataset('value', data=img);
            fOut.close()

            #== Note: normalisation - temporarily exempted ==#
            #rangeOut = open(os.path.join(trainOutputDir,fname+"_norm"+".txt"),"w")
            #rangeOut.write("min_global: %g\n" % minVal)
            #rangeOut.write("max_global: %g" % maxVal)
            #rangeOut.close()
        del img

        f = h5py.File(scatterName, 'r')
        img = numpy.array(f['dataY'], order='F').transpose()
        f.close()
        #img = numpy.squeeze(numpy.sum(img, 2))
        #== Note: to be done on data load to reduce load-from-disk time ==#
        #img = ndimage.zoom(img, [SCALE_FACTOR,SCALE_FACTOR,1], order=2)
        #== Note: normalisation - temporarily exempted ==#
        #minVal = numpy.min(img)
        #maxVal = numpy.max(img)
        #img = (img-minVal)/(maxVal-minVal)
        #== Note: for over-sized dataset: crop here ==#
        if img.shape[0] > 150:
            if prtsz == True:
                print(img.shape)
            im_center = numpy.array([(img.shape[0]-1)/2, (img.shape[1]-1)/2], dtype=numpy.int32)
            left = im_center[0]-74
            right = left+150
            top = im_center[1]-74
            bottom = top+150
            img = img[left:right,top:bottom,:]
            if prtsz == True:
                print(img.shape)
                prtsz = False
        if j<TEST_LENGTH:
            fname = "%s_%05d_Y" % (options.datasetName,global_data_index)
            fOut = h5py.File(os.path.join(testOutputDir,fname+".h5"),'w')
            outGroup = fOut.create_group('data')
            #outGroup.create_dataset('value', data=img.astype(numpy.float32));
            outGroup.create_dataset('value', data=img);
            fOut.close()

            #== Note: normalisation - temporarily exempted ==#
            #rangeOut = open(os.path.join(trainOutputDir,fname+"_norm"+".txt"),"w")
            #rangeOut.write("min_global: %g\n" % minVal)
            #rangeOut.write("max_global: %g" % maxVal)
            #rangeOut.close()
        del img

        f = h5py.File(observeName, 'r')
        img = numpy.array(f['dataZ'], order='F').transpose()
        f.close()
        #img = numpy.squeeze(numpy.sum(img, 2))
        #== Note: to be done on data load to reduce load-from-disk time ==#
        #img = ndimage.zoom(img, [SCALE_FACTOR,SCALE_FACTOR,1], order=2)
        #== Note: normalisation - temporarily exempted ==#
        #minVal = numpy.min(img)
        #maxVal = numpy.max(img)
        #img = (img-minVal)/(maxVal-minVal)
        #== Note: for over-sized dataset: crop here ==#
        if img.shape[0] > 150:
            if prtsz == True:
                print(img.shape)
            im_center = numpy.array([(img.shape[0]-1)/2, (img.shape[1]-1)/2], dtype=numpy.int32)
            left = im_center[0]-74
            right = left+150
            top = im_center[1]-74
            bottom = top+150
            img = img[left:right,top:bottom,:]
            if prtsz == True:
                print(img.shape)
                prtsz = False
        if j<TEST_LENGTH:
            fname = "%s_%05d_Z" % (options.datasetName,global_data_index)
            fOut = h5py.File(os.path.join(testOutputDir,fname+".h5"),'w')
            outGroup = fOut.create_group('data')
            #outGroup.create_dataset('value', data=img.astype(numpy.float32));
            outGroup.create_dataset('value', data=img);
            fOut.close()

            #== Note: normalisation - temporarily exempted ==#
            #rangeOut = open(os.path.join(trainOutputDir,fname+"_norm"+".txt"),"w")
            #rangeOut.write("min_global: %g\n" % minVal)
            #rangeOut.write("max_global: %g" % maxVal)
            #rangeOut.close()
        del img

        if j<TEST_LENGTH:
            #fname = "%s_%05d" % (options.datasetName,global_data_index)
            #print("processed entry '%s'" % (fname))
            if ((global_data_index%100) == 0) and (global_data_index!=0):
                print("progress: %02f %%" % (global_data_index*100/lenX))
            global_data_index+=1
        j+=1

    #print("Input range: %s" % (str(inputRange)))
    #print("Scatter map range: %s" % (str(scatterRange)))
    #print("Observation range: %s" % (str(observationRange)))
    
    #inputData = (inputData-inputRange[0])/(inputRange[1]-inputRange[0])
    #inputData_validation = (inputData_validation-inputRange[0])/(inputRange[1]-inputRange[0])
    #inputData_test = (inputData_test-inputRange[0])/(inputRange[1]-inputRange[0])
    #observationData = (observationData-observationRange[0])/(observationRange[1]-observationRange[0])
    #observationData_validation = (observationData_validation-observationRange[0])/(observationRange[1]-observationRange[0])
    #observationData_test = (observationData_test-observationRange[0])/(observationRange[1]-observationRange[0])
