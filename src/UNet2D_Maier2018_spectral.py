from __future__ import print_function

import numpy
from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Add, SeparableConv2D, Dense, Lambda
from keras.layers import UpSampling2D, Dropout, BatchNormalization, Reshape, concatenate


class UNet2D_Maier2018_spectral(object):
    """UNetFactory helps create a UNet model using Keras."""

    def __init__(self):
        super(UNet2D_Maier2018_spectral, self).__init__()
        self._activation = 'relu'
        self.dropout=True

        #self.dataShape = None
        self.input = None  # Holder for Model input
        self.output = None  # Holder for Model output

    def begin(self, input_shape, output_shape=None):
        """
        Input here is an image of the forward scatter intensity
        of the measured data (c_0=1); In imaging terms, that's a 1-channel 2D image.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        print("input shape {}".format(input_shape))
        self.input = Input(shape=input_shape)

    def finalize(self):
        """
        The expected output data is the scatter prediction; in train-test scenarios,
        this is the Monte Carlo simulation of the scattering.
        The output variable holds the network to achieve that.
        The function returns the model itself.
        """
        if self.input is None:
            raise RuntimeError("Missing an input. Use begin().")
        if self.output is None:
            raise RuntimeError("Missing an output. Use buildNetwork().")
        return Model(inputs=self.input, outputs=self.output)
    
    def buildNetwork(self):
        """
        This function builds the network exactly as-is from the paper of Maier et al.
        """
        #nFilterArray = [40,80,160,320,480,960]
        #nFilterArray = [40*16, 80*16, 160*16, 320*16, 480*16, 960*16]   # try with 64x64x16 image sample
        #nFilterArray = [40 * 8, 80 * 8, 160 * 8, 320 * 8]  # try with 64x64x16 image sample
        
        #=== arch1 ===#
        #nFilterArray = [48,64,128,256,512,1024] # nFeat: 52776558133248;  not efficient: too few features on pixel level, too many high-level features;
        #=== archX ===#
        #nFilterArray = [64,128,256,384,576,864]  # nFeat: 400771988324352; too many features (most closely resembles original network) [tested]
        #=== arch4 ===#
        #nFilterArray = [64,128,192,288,576,1152] # nFeat: 300578991243264; [working]
        #=== arch5 ===#
        #nFilterArray = [256,256,256,256,256,256] # nFeat: 281474976710656; [high number of starting features, TBT]
        #=== arch3 ===#
        #nFilterArray = [48,96,192,384,576,864]   # nFeat: 169075682574336; (working) [tested]
        #=== arch2 ===#
        #nFilterArray = [48,96,144,288,432,864]   # nFeat: 71328803586048;  (working), too few features
        print("Create input layer with shape {}".format(self.input.shape))

        #numInChannels = self.input.shape[len(self.input.shape)-1]
        numInChannels = self.input_shape[2]
        individual_outputs = []
        for i in range(0,numInChannels):
            #nFilterArray = [40, 80, 160, 320, 480, 960]
            nFilterArray = [40, 80, 160, 320,480]
            #in_data = self.input[:,:,:,i]
            #in_data_t =  Reshape((self.input_shape[0], self.input_shape[1], 1))(in_data)
            in_data = Lambda(lambda x: x[:,:,:, i])(self.input) #output_shape=self.input_shape[:-1]+(1,)
            in_data_t =  Reshape((self.input_shape[0], self.input_shape[1], 1))(in_data)
            #print(in_data_t.shape)
            individual_shape = in_data_t.shape
            individual_outputs.append(self._createChannelLayer(in_data_t,i,nFilterArray,None))
        #numOutChannels = individualOutputs[0].shape[len(individualOutputs[0].shape)-1]
        numOutChannels = len(individual_outputs)
        if numOutChannels!=numInChannels:
            print("Channels do not match (in: {} vs out: {}.".format(numInChannels,numOutChannels))
        self.output = concatenate(individual_outputs) #len(individualOutputs[0].shape)-1

        #self.output = SeparableConv2D(self.dataShape[2], 1, depth_multiplier=1, activation=self._activation, padding='same')(m)
        # "residual learning", or rather: learn the corrected image, not the correction step.
        #self.output = Add()([self.input, self.output])
        print("Created output layer with shape {}".format(self.output.shape))
        #exit()
        # ================== #        

    def _createChannelLayer(self, in_data, channelId, filterBankArray, inShapePerImage):
        num_total = len(filterBankArray)
        # START INPUT LAYER #
        nFilters = 4
        convSize = (3, 3)
        if (inShapePerImage is not None):
            n = self._getConv2D_(in_data, nFilters, convSize, shp=inShapePerImage, dFormat="channels_last")
            # n = SeparableConv2D(nFilters,convSize,depth_multiplier=16,activation=self._activation, padding='same', input_shape=inShapePerImage, data_format="channels_last")(self.input)
        else:
            n = self._getConv2D_(in_data, nFilters, convSize)
            # n = SeparableConv2D(nFilters, convSize, depth_multiplier=16, activation=self._activation, padding='same')(self.input)
        # n = Dense(nFilters*8, activation=self._activation)(n)
        #print("Created layer {} with shape {}".format(0, n.shape))
        m = MaxPooling2D()(n)
        # ================= #
        m = self._getRecursiveLayer_(m, 0, num_total, filterBankArray, convSize)
        # FINAL OUTPUT LAYER #
        m = UpSampling2D()(m)
        m = self._getConv2D_(m, 20, convSize)
        m = Concatenate(axis=3)([n, m])
        m = self._getConv2D_(m, 20, convSize)
        m = self._getConv2D_(m, 20, convSize)
        #print("Created layer {} with shape {}".format(0, m.shape))
        # self.output = self._getConv2D_(m, 1, (1,1))        # standard procedure for last-layer convolution
        # self.output = Conv2D(self.dataShape[2], 1)(m)
        # self.output = SeparableConv2D(self.dataShape[2], 1, depth_multiplier=1, activation=self._activation, padding='same')(m)
        #out_data = self._getConv2D_(m, 1, (1, 1))
        out_data = Conv2D(1,1)(m)
        #print("Created individual output layer with shape {}".format(out_data.shape))
        return out_data

    def _getRecursiveLayer_(self, previousNetwork, num, numTotal, filterBankArray, convSize):
        nFilters = filterBankArray[0]
        if(len(filterBankArray)>1):
            nFilterArray = filterBankArray[1:]
        if num < (numTotal-1):  # If not, we don't need more sublevels.
            n = self._getConv2D_(previousNetwork, nFilters, convSize)
            n = Dropout(0.005)(n) if self.dropout else n
            n = self._getConv2D_(n, nFilters, convSize)
            #print("Created layer {} with shape {}".format(num+1, n.shape))
            m = MaxPooling2D()(n)
            
            
            num = num+1
            m = self._getRecursiveLayer_(m, num, numTotal, nFilterArray, convSize)
            
            m = UpSampling2D()(m)
            #m = self._getConv2D_(m, nFilters, convSize)    # not in Maier et al- 2018
            m = Concatenate(axis=3)([n, m])
        else:
            m = previousNetwork
        m = self._getConv2D_(m, nFilters, convSize)
        m = self._getConv2D_(m, nFilters, convSize)
        #print("Created layer {} with shape {}".format(num+1, m.shape))
        return m
    
    def _getConv2D_(self, prevIn, nFilters, convSize, shp = None, dFormat=None):
        """Returns a convolution."""
        if (shp is not None) and (dFormat is not None):
            m = Conv2D(nFilters, convSize, activation=self._activation, padding='same', input_shape=shp, data_format=dFormat)(prevIn)
            return m
        m = Conv2D(nFilters, convSize, activation=self._activation, padding='same')(prevIn)
        return m
