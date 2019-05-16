from __future__ import print_function

from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Add
from keras.layers import UpSampling2D, Dropout, BatchNormalization


class UNet2D_Maier2018(object):
    """UNetFactory helps create a UNet model using Keras."""

    def __init__(self):
        super(UNet2D_Maier2018, self).__init__()
        self._activation = 'relu'
        self.dropout=True

        self.input = None  # Holder for Model input
        self.output = None  # Holder for Model output
        
    def begin(self, image_shape):
        """
        Input here is an image of the forward scatter intensity
        of the measured data (c_0=1); In imaging terms, that's a 1-channel 2D image.
        """
        print("Created input layer with shape {}".format(image_shape))
        self.input = Input(shape=image_shape)
    
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
    
    def buildNetwork(self, inShapePerImage=None):
        """
        This function builds the network exactly as-is from the paper of Maier et al.
        """
        nFilterArray = [40,80,160,320,480,960]
        print("Created input layer with shape {}".format(self.input.shape))
        num_total = len(nFilterArray)
        # START INPUT LAYER #
        nFilters = 4
        #nFilters = 1
        convSize = (3,3)
        if(inShapePerImage!=None):
            n = self._getConv2D_(self.input, nFilters, convSize, shp=inShapePerImage, dFormat="channels_last")
        else:
            n = self._getConv2D_(self.input, nFilters, convSize)
        print("Created layer {} with shape {}".format(0, n.shape))
        m = MaxPooling2D()(n)
        # ================= #
        
        m = self._getRecursiveLayer_(m, 0, num_total, nFilterArray, convSize)

        # FINAL OUTPUT LAYER #
        m = UpSampling2D()(m)
        m = self._getConv2D_(m, 20, convSize)
        m = Concatenate(axis=3)([n, m])
        m = self._getConv2D_(m, 20, convSize)
        m = self._getConv2D_(m, 20, convSize)
        print("Created layer {} with shape {}".format(0, m.shape))
        #self.output = self._getConv2D_(m, 1, (1,1))        # standard procedure for last-layer convolution
        self.output = Conv2D(1, 1)(m)
        print("Created output layer with shape {}".format(self.output.shape))
        # ================== #        
        
    def _getRecursiveLayer_(self, previousNetwork, num, numTotal, nFilterArray, convSize):
        nFilters = nFilterArray[0]
        if(len(nFilterArray)>1):
            nFilterArray = nFilterArray[1:]
        if num < (numTotal-1):  # If not, we don't need more sublevels.
            n = self._getConv2D_(previousNetwork, nFilters, convSize)
            n = Dropout(0.005)(n) if self.dropout else n
            n = self._getConv2D_(n, nFilters, convSize)
            print("Created layer {} with shape {}".format(num+1, n.shape))
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
        print("Created layer {} with shape {}".format(num+1, m.shape)) 
        return m
    
    def _getConv2D_(self, prevIn, nFilters, convSize, shp = None, dFormat=None):
        """Returns a convolution."""
        if (shp!=None) and (dFormat!=None):
            m = Conv2D(nFilters, convSize, activation=self._activation, padding='same', input_shape=shp, data_format=dFormat)(prevIn)
            return m
        m = Conv2D(nFilters, convSize, activation=self._activation, padding='same')(prevIn)
        return m