from __future__ import print_function

from keras.models import Input, Model
from keras.layers import Dropout, BatchNormalization, Concatenate, Add, Reshape, concatenate
from keras.layers import UpSampling1D, Conv1D, MaxPooling1D, UpSampling2D, Conv2D, MaxPooling2D, UpSampling3D, Conv3D, MaxPooling3D
from keras import regularizers

class UNetFactory(object):
    """UNetFactory helps create a UNet model using Keras."""

    def __init__(self):
        super(UNetFactory, self).__init__()

        self.conv_k_1D = (3,)
        self.conv_k_2D = (3,3)
        self.conv_k_3D = (3,3,5)
        #self.increment_filter_rate = 1
        self.activ_func = 'relu'
        self.dropout = 0.02
        self.batch_norm = False
        self.input_shape = (1,1,1,1)
        #self.reg = regularizers.l2(0.01)
        self.reg=None

        self.input = None  # Holder for Model inpuT; COMMON SIZE: 100X100X32(X1)
        self.output = None  # Holder for Model output

    def begin(self, image_shape):
        self.input_shape = image_shape
        self.input = Input(shape=self.input_shape)

    def buildNetwork(self):
        print("Create input layer with shape {} (expected: {})".format(self.input.shape, self.input_shape))
        net1D = self.build_1Dnetwork(self.input)
        print("Create 1D network with shape {}".format(net1D.shape))
        net2D = self.build_2Dnetwork(self.input)
        print("Create 2D network with shape {}".format(net2D.shape))
        net3D = self.build_3Dnetwork(self.input)
        print("Create 3D network with shape {}".format(net3D.shape))
        fuseNets = concatenate([net1D, net2D, net3D])
        print("Create fused network with shape {}".format(fuseNets.shape))
        #conv3D_fuse_1 = Conv3D(8, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(fuseNets)
        #print("Create fused-conv 1 network with shape {}".format(conv3D_fuse_1.shape))
        #conv3D_fuse_2 = Conv3D(8, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv3D_fuse_1)
        #print("Create fused-conv 2 network with shape {}".format(conv3D_fuse_2.shape))
        conv3D_out = Conv3D(1, (1,1,1), activation=self.activ_func, padding="same", activity_regularizer=self.reg)(fuseNets)
        print("Create output conv network with shape {}".format(conv3D_out.shape))
        #self.output = Add()([self.input, conv3D_out])
        self.output = conv3D_out
        
        #fuseNets = concatenate([net1D, net2D, net3D], axis=-2)
        #print("Create fused network with shape {}".format(fuseNets.shape))
        #shape_conv2D = (self.input_shape[0], self.input_shape[1], self.input_shape[2]*self.input_shape[3]*3)
        #conv2D_in_reshape = Reshape(shape_conv2D)(fuseNets)
        #conv2D_fuse_1 = Conv2D(2*self.input_shape[2]*self.input_shape[3], self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_in_reshape)
        #print("Create fused-conv 1 network with shape {}".format(conv2D_fuse_1.shape))
        #conv2D_fuse_2 = Conv2D(1*self.input_shape[2]*self.input_shape[3], self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_fuse_1)
        #print("Create fused-conv 2 network with shape {}".format(conv2D_fuse_2.shape))
        #conv2D_out_reshape = Reshape(self.input_shape)(conv2D_fuse_2)
        #self.output = Add()([self.input, conv2D_out_reshape])

        return Model(inputs=self.input, outputs=self.output)


    def build_1Dnetwork(self, input_data):
        # == Input reshape == #
        shape_conv1D = (self.input_shape[0]*self.input_shape[1], self.input_shape[2]*self.input_shape[3])
        conv1D_in_reshape = Reshape(shape_conv1D)(input_data)
        # == Layer 1 - down == #
        conv1D_l1d_1 = Conv1D(4, self.conv_k_1D, activation=self.activ_func, input_shape=shape_conv1D, data_format="channels_last", padding="same", activity_regularizer=self.reg)(conv1D_in_reshape)
        conv1D_l1d_2 = Conv1D(4, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv1D_l1d_1)
        #dp1D_l1d_3 = Dropout(self.dropout)(conv1D_l1d_2)
        mxp1D_l1d_4 = MaxPooling1D(pool_size=2)(conv1D_l1d_2)
        # == Layer 2 - down == #
        conv1D_l2d_1 = Conv1D(8, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp1D_l1d_4)
        conv1D_l2d_2 = Conv1D(8, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv1D_l2d_1)
        #dp1D_l2d_3 = Dropout(self.dropout)(conv1D_l2d_2)
        mxp1D_l2d_4 = MaxPooling1D(pool_size=4)(conv1D_l2d_2)
        # == Layer 3 - bottom == #
        conv1D_l3_1 = Conv1D(16, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp1D_l2d_4)
        conv1D_l3_2 = Conv1D(16, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv1D_l3_1)
        #dp1D_l3_3 = Dropout(self.dropout)(conv1D_l3_2)
        ups1D_l3_4 = UpSampling1D(size=4)(conv1D_l3_2)
        # == Layer 2 - up == #
        conv1D_l2s_1 = Conv1D(8, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups1D_l3_4)
        concat1D_l2s_2 = Concatenate(axis=2)([conv1D_l2d_2, conv1D_l2s_1])
        conv1D_l2s_3 = Conv1D(8, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat1D_l2s_2)
        conv1D_l2s_4 = Conv1D(8, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv1D_l2s_3)
        ups1D_l3_5 = UpSampling1D(size=2)(conv1D_l2s_4)
        # == Layer 1 - up == #
        conv1D_l1s_1 = Conv1D(4, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups1D_l3_5)
        concat1D_l1s_2 = Concatenate(axis=2)([conv1D_l1d_2, conv1D_l1s_1])
        conv1D_l1s_3 = Conv1D(4, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat1D_l1s_2)
        conv1D_l1s_4 = Conv1D(4, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv1D_l1s_3)
        # == Output layer == #
        conv1D_out_1 = Conv1D(self.input_shape[2]*self.input_shape[3], (1,), activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv1D_l1s_4)
        conv1D_out_reshape = Reshape(self.input_shape)(conv1D_out_1)
        return conv1D_out_reshape

    def build_2Dnetwork(self, input_data):
         # == Input reshape == #
         shape_conv2D = (self.input_shape[0], self.input_shape[1], self.input_shape[2]*self.input_shape[3])
         conv2D_in_reshape = Reshape(shape_conv2D)(input_data)
         # == Layer 1 - down == #
         conv2D_l1d_1 = Conv2D(4, self.conv_k_2D, activation=self.activ_func, input_shape=shape_conv2D, data_format="channels_last", padding="same", activity_regularizer=self.reg)(conv2D_in_reshape)
         conv2D_l1d_2 = Conv2D(4, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l1d_1)
         dp2D_l1d_3 = Dropout(self.dropout)(conv2D_l1d_2)
         mxp2D_l1d_4 = MaxPooling2D()(dp2D_l1d_3)
         # == Layer 2 - down == #
         conv2D_l2d_1 = Conv2D(8, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp2D_l1d_4)
         conv2D_l2d_2 = Conv2D(8, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l2d_1)
         dp2D_l2d_3 = Dropout(self.dropout)(conv2D_l2d_2)
         mxp2D_l2d_4 = MaxPooling2D()(dp2D_l2d_3)
         # == Layer 3 - down == #
         conv2D_l3d_1 = Conv2D(16, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp2D_l2d_4)
         conv2D_l3d_2 = Conv2D(16, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l3d_1)
         dp2D_l3d_3 = Dropout(self.dropout)(conv2D_l3d_2)
         mxp2D_l3d_4 = MaxPooling2D()(dp2D_l3d_3)
 
         ## == Layer 4 - down == #
         conv2D_l4d_1 = Conv2D(32, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp2D_l3d_4)
         conv2D_l4d_2 = Conv2D(32, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l4d_1)
         dp2D_l4d_3 = Dropout(self.dropout)(conv2D_l4d_2)
         mxp2D_l4d_4 = MaxPooling2D()(dp2D_l4d_3)
 
         # == Layer 5 - bottom == #
         conv2D_l5_1 = Conv2D(64, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp2D_l4d_4)
         conv2D_l5_2 = Conv2D(64, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l5_1)
#         dp2D_l5_3 = Dropout(self.dropout)(conv2D_l5_2)
         ups2D_l5_4 = UpSampling2D()(conv2D_l5_2)
 
         # == Layer 4 - up == #
         conv2D_l4s_1 = Conv2D(32, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups2D_l5_4)
         concat2D_l4s_2 = Concatenate(axis=3)([dp2D_l4d_3, conv2D_l4s_1])
         conv2D_l4s_3 = Conv2D(32, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat2D_l4s_2)
         conv2D_l4s_4 = Conv2D(32, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l4s_3)
         ups2D_l4s_5 = UpSampling2D()(conv2D_l4s_4)
 
         # == Layer 3 - up == #
         conv2D_l3s_1 = Conv2D(16, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups2D_l4s_5)
         concat2D_l3s_2 = Concatenate(axis=3)([dp2D_l3d_3, conv2D_l3s_1])
         conv2D_l3s_3 = Conv2D(16, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat2D_l3s_2)
         conv2D_l3s_4 = Conv2D(16, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l3s_3)
         ups2D_l3s_5 = UpSampling2D()(conv2D_l3s_4)
         # == Layer 2 - up == #
         conv2D_l2s_1 = Conv2D(8, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups2D_l3s_5)
         concat2D_l2s_2 = Concatenate(axis=3)([dp2D_l2d_3, conv2D_l2s_1])
         conv2D_l2s_3 = Conv2D(8, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat2D_l2s_2)
         conv2D_l2s_4 = Conv2D(8, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l2s_3)
         ups2D_l2s_5 = UpSampling2D()(conv2D_l2s_4)
         # == Layer 1 - up == #
         conv2D_l1s_1 = Conv2D(4, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups2D_l2s_5)
         concat2D_l1s_2 = Concatenate(axis=3)([dp2D_l1d_3, conv2D_l1s_1])
         conv2D_l1s_3 = Conv2D(4, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat2D_l1s_2)
         conv2D_l1s_4 = Conv2D(4, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l1s_3)
         # == Output layer == #
         conv2D_out_1 = Conv2D(self.input_shape[2]*self.input_shape[3], (1,1), activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l1s_4)
         conv2D_out_reshape = Reshape(self.input_shape)(conv2D_out_1)
         return conv2D_out_reshape
 
    def build_3Dnetwork(self, input_data):
        # == Input reshape == #
        shape_conv3D = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
        conv3D_in_reshape = Reshape(shape_conv3D)(input_data)
        # == Layer 1 - down == #
        conv3D_l1d_1 = Conv3D(4, self.conv_k_3D, activation=self.activ_func, input_shape=shape_conv3D, data_format="channels_last", padding="same", activity_regularizer=self.reg)(conv3D_in_reshape)
        conv3D_l1d_2 = Conv3D(4, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv3D_l1d_1)
        dp3D_l1d_3 = Dropout(self.dropout)(conv3D_l1d_2)
        mxp3D_l1d_4 = MaxPooling3D(pool_size=(4,4,4))(dp3D_l1d_3)   # (256x256x8) -> (64x64x2)
        # == Layer 2 - bottom == #
        conv3D_l2d_1 = Conv3D(8, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp3D_l1d_4)
        conv3D_l2d_2 = Conv3D(8, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv3D_l2d_1)
        dp3D_l2d_3 = Dropout(self.dropout)(conv3D_l2d_2)
        mxp3D_l2d_4 = MaxPooling3D(pool_size=(2,2,2))(dp3D_l2d_3)   # (64x64x2) -> (32x32x1)
        # == Layer 3 - bottom == #
        conv3D_l3_1 = Conv3D(16, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp3D_l2d_4)
        conv3D_l3_2 = Conv3D(16, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv3D_l3_1)
        ups3D_l3_3 = UpSampling3D(size=(2,2,2))(conv3D_l3_2)
        # == Layer 2 - up == #
        conv3D_l2s_1 = Conv3D(8, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups3D_l3_3)
        concat3D_l2s_2 = Concatenate(axis=4)([dp3D_l2d_3, conv3D_l2s_1])
        conv3D_l2s_3 = Conv3D(8, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat3D_l2s_2)
        conv3D_l2s_4 = Conv3D(8, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv3D_l2s_3)
        ups3D_l2s_5 = UpSampling3D(size=(4,4,4))(conv3D_l2s_4)
        # == Layer 1 - up == #
        conv3D_l1s_1 = Conv3D(4, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups3D_l2s_5)
        concat3D_l1s_2 = Concatenate(axis=4)([dp3D_l1d_3, conv3D_l1s_1])
        conv3D_l1s_2 = Conv3D(4, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat3D_l1s_2)
        conv3D_l1s_3 = Conv3D(4, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv3D_l1s_2)
        # == Output layer == #
        conv3D_out_1 = Conv3D(1, (1,1,1), activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv3D_l1s_3)
        conv3D_out_reshape = Reshape(self.input_shape)(conv3D_out_1)
        return conv3D_out_reshape
    
# =============================================================================
#     def build_1Dnetwork(self, input_data):
#         # == Input reshape == #
#         shape_conv1D = (self.input_shape[0]*self.input_shape[1], self.input_shape[2]*self.input_shape[3])
#         conv1D_in_reshape = Reshape(shape_conv1D)(input_data)
#         # == Layer 1 - down == #
#         conv1D_l1d_1 = Conv1D(16, self.conv_k_1D, activation=self.activ_func, input_shape=shape_conv1D, data_format="channels_last", padding="same", activity_regularizer=self.reg)(conv1D_in_reshape)
#         conv1D_l1d_2 = Conv1D(16, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv1D_l1d_1)
#         #dp1D_l1d_3 = Dropout(self.dropout)(conv1D_l1d_2)
#         mxp1D_l1d_4 = MaxPooling1D(pool_size=2)(conv1D_l1d_2)
#         # == Layer 2 - down == #
#         conv1D_l2d_1 = Conv1D(32, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp1D_l1d_4)
#         conv1D_l2d_2 = Conv1D(32, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv1D_l2d_1)
#         #dp1D_l2d_3 = Dropout(self.dropout)(conv1D_l2d_2)
#         mxp1D_l2d_4 = MaxPooling1D(pool_size=4)(conv1D_l2d_2)
#         # == Layer 3 - bottom == #
#         conv1D_l3_1 = Conv1D(64, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp1D_l2d_4)
#         conv1D_l3_2 = Conv1D(64, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv1D_l3_1)
#         #dp1D_l3_3 = Dropout(self.dropout)(conv1D_l3_2)
#         ups1D_l3_4 = UpSampling1D(size=4)(conv1D_l3_2)
#         # == Layer 2 - up == #
#         conv1D_l2s_1 = Conv1D(32, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups1D_l3_4)
#         concat1D_l2s_2 = Concatenate(axis=2)([conv1D_l2d_2, conv1D_l2s_1])
#         conv1D_l2s_3 = Conv1D(32, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat1D_l2s_2)
#         conv1D_l2s_4 = Conv1D(32, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv1D_l2s_3)
#         ups1D_l3_5 = UpSampling1D(size=2)(conv1D_l2s_4)
#         # == Layer 1 - up == #
#         conv1D_l1s_1 = Conv1D(16, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups1D_l3_5)
#         concat1D_l1s_2 = Concatenate(axis=2)([conv1D_l1d_2, conv1D_l1s_1])
#         conv1D_l1s_3 = Conv1D(16, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat1D_l1s_2)
#         conv1D_l1s_4 = Conv1D(16, self.conv_k_1D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv1D_l1s_3)
#         # == Output layer == #
#         conv1D_out_1 = Conv1D(self.input_shape[2]*self.input_shape[3], (1,), activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv1D_l1s_4)
#         conv1D_out_reshape = Reshape(self.input_shape)(conv1D_out_1)
#         return conv1D_out_reshape    
# =============================================================================

# =============================================================================
#     def build_2Dnetwork(self, input_data):
#         # == Input reshape == #
#         shape_conv2D = (self.input_shape[0], self.input_shape[1], self.input_shape[2]*self.input_shape[3])
#         conv2D_in_reshape = Reshape(shape_conv2D)(input_data)
#         # == Layer 1 - down == #
#         conv2D_l1d_1 = Conv2D(16, self.conv_k_2D, activation=self.activ_func, input_shape=shape_conv2D, data_format="channels_last", padding="same", activity_regularizer=self.reg)(conv2D_in_reshape)
#         conv2D_l1d_2 = Conv2D(16, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l1d_1)
#         dp2D_l1d_3 = Dropout(self.dropout)(conv2D_l1d_2)
#         mxp2D_l1d_4 = MaxPooling2D()(dp2D_l1d_3)
#         # == Layer 2 - down == #
#         conv2D_l2d_1 = Conv2D(32, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp2D_l1d_4)
#         conv2D_l2d_2 = Conv2D(32, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l2d_1)
#         dp2D_l2d_3 = Dropout(self.dropout)(conv2D_l2d_2)
#         mxp2D_l2d_4 = MaxPooling2D()(dp2D_l2d_3)
#         # == Layer 3 - down == #
#         conv2D_l3d_1 = Conv2D(48, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp2D_l2d_4)
#         conv2D_l3d_2 = Conv2D(48, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l3d_1)
#         dp2D_l3d_3 = Dropout(self.dropout)(conv2D_l3d_2)
#         mxp2D_l3d_4 = MaxPooling2D()(dp2D_l3d_3)
# 
#         ## == Layer 4 - down == #
#         #conv2D_l4d_1 = Conv2D(64, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp2D_l3d_4)
#         #conv2D_l4d_2 = Conv2D(64, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l4d_1)
#         #dp2D_l4d_3 = Dropout(self.dropout)(conv2D_l4d_2)
#         #mxp2D_l4d_4 = MaxPooling2D()(dp2D_l4d_3)
# 
#         # == Layer 5 - bottom == #
#         conv2D_l5_1 = Conv2D(80, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp2D_l3d_4)
#         conv2D_l5_2 = Conv2D(80, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l5_1)
#         #dp2D_l5_3 = Dropout(self.dropout)(conv2D_l5_2)
#         ups2D_l5_4 = UpSampling2D()(conv2D_l5_2)
# 
#         # == Layer 4 - up == #
#         #conv2D_l4s_1 = Conv2D(64, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups2D_l5_4)
#         #concat2D_l4s_2 = Concatenate(axis=3)([dp2D_l4d_3, conv2D_l4s_1])
#         #conv2D_l4s_3 = Conv2D(64, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat2D_l4s_2)
#         #conv2D_l4s_4 = Conv2D(64, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l4s_3)
#         #ups2D_l4s_5 = UpSampling2D()(conv2D_l4s_4)
# 
#         # == Layer 3 - up == #
#         conv2D_l3s_1 = Conv2D(48, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups2D_l5_4)
#         concat2D_l3s_2 = Concatenate(axis=3)([dp2D_l3d_3, conv2D_l3s_1])
#         conv2D_l3s_3 = Conv2D(48, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat2D_l3s_2)
#         conv2D_l3s_4 = Conv2D(48, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l3s_3)
#         ups2D_l3s_5 = UpSampling2D()(conv2D_l3s_4)
#         # == Layer 2 - up == #
#         conv2D_l2s_1 = Conv2D(32, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups2D_l3s_5)
#         concat2D_l2s_2 = Concatenate(axis=3)([dp2D_l2d_3, conv2D_l2s_1])
#         conv2D_l2s_3 = Conv2D(32, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat2D_l2s_2)
#         conv2D_l2s_4 = Conv2D(32, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l2s_3)
#         ups2D_l2s_5 = UpSampling2D()(conv2D_l2s_4)
#         # == Layer 1 - up == #
#         conv2D_l1s_1 = Conv2D(16, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups2D_l2s_5)
#         concat2D_l1s_2 = Concatenate(axis=3)([dp2D_l1d_3, conv2D_l1s_1])
#         conv2D_l1s_3 = Conv2D(16, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat2D_l1s_2)
#         conv2D_l1s_4 = Conv2D(16, self.conv_k_2D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l1s_3)
#         # == Output layer == #
#         conv2D_out_1 = Conv2D(self.input_shape[2]*self.input_shape[3], (1,1), activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv2D_l1s_4)
#         conv2D_out_reshape = Reshape(self.input_shape)(conv2D_out_1)
#         return conv2D_out_reshape
# =============================================================================
        

# =============================================================================
#     def build_3Dnetwork(self, input_data):
#         # == Input reshape == #
#         shape_conv3D = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
#         conv3D_in_reshape = Reshape(shape_conv3D)(input_data)
#         # == Layer 1 - down == #
#         conv3D_l1d_1 = Conv3D(4, self.conv_k_3D, activation=self.activ_func, input_shape=shape_conv3D, data_format="channels_last", padding="same", activity_regularizer=self.reg)(conv3D_in_reshape)
#         conv3D_l1d_2 = Conv3D(4, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv3D_l1d_1)
#         dp3D_l1d_3 = Dropout(self.dropout)(conv3D_l1d_2)
#         mxp3D_l1d_4 = MaxPooling3D(pool_size=(4,4,4))(dp3D_l1d_3)
#         # == Layer 2 - bottom == #
#         conv3D_l2d_1 = Conv3D(8, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp3D_l1d_4)
#         conv3D_l2d_2 = Conv3D(8, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv3D_l2d_1)
#         dp3D_l2d_3 = Dropout(self.dropout)(conv3D_l2d_2)
#         mxp3D_l2d_4 = MaxPooling3D(pool_size=(2, 2, 2))(dp3D_l2d_3)
#         # == Layer 3 - bottom == #
#         conv3D_l3_1 = Conv3D(16, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(mxp3D_l2d_4)
#         conv3D_l3_2 = Conv3D(16, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv3D_l3_1)
#         ups3D_l3_3 = UpSampling3D(size=(2,2,2))(conv3D_l3_2)
#         # == Layer 2 - up == #
#         conv3D_l2s_1 = Conv3D(8, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups3D_l3_3)
#         concat3D_l2s_2 = Concatenate(axis=4)([dp3D_l2d_3, conv3D_l2s_1])
#         conv3D_l2s_3 = Conv3D(8, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat3D_l2s_2)
#         conv3D_l2s_4 = Conv3D(8, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv3D_l2s_3)
#         ups3D_l2s_5 = UpSampling3D(size=(4,4,4))(conv3D_l2s_4)
#         # == Layer 1 - up == #
#         conv3D_l1s_1 = Conv3D(4, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(ups3D_l2s_5)
#         concat3D_l1s_2 = Concatenate(axis=4)([dp3D_l1d_3, conv3D_l1s_1])
#         conv3D_l1s_2 = Conv3D(4, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(concat3D_l1s_2)
#         conv3D_l1s_3 = Conv3D(4, self.conv_k_3D, activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv3D_l1s_2)
#         # == Output layer == #
#         conv3D_out_1 = Conv3D(1, (1,1,1), activation=self.activ_func, padding="same", activity_regularizer=self.reg)(conv3D_l1s_3)
#         conv3D_out_reshape = Reshape(self.input_shape)(conv3D_out_1)
#         return conv3D_out_reshape    
# =============================================================================
