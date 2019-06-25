#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 19 2018

@author: chke (Christian Kehl)
"""

import numpy
import sys
import itertools
from multiprocessing import Lock,RLock,Semaphore,Event,Array,Value
from multiprocessing.managers import BaseManager
import ctypes
import math

WORKERS = 12
#CACHE_SIZE = 32
CACHE_SIZE = 128
CACHE_REUSE_PERIOD=12288  # should be at least WORKERS * CACHE_SIZE

class MemoryCache(object):


    def __init__(self):
        self.cacheX = None
        self.cacheX_base = None
        self.limitsX = None
        self.limitsX_base = None
        self.cache_used_counter_x = Value('i', 0)
        #self.cache_renewed_counter_x = Value('i', 0)
        self.image_shape_x = None
        self.cache_shape_x = None
        self.numE_x = None
        self.cacheY = None
        self.cacheY_base = None
        self.limitsY = None
        self.limitsY_base = None
        self.cache_used_counter_y = Value('i', 0)
        #self.cache_renewed_counter_y = Value('i', 0)
        self.image_shape_y = None
        self.cache_shape_y = None
        self.numE_y = None
        self.cache_size = CACHE_SIZE
        self.cache_period = CACHE_REUSE_PERIOD
        self.renew_cache = Value(ctypes.c_bool, False)
        self.cache_used_counter = Value('i', 0)
        self.cache_renewed_counter = Value('i', 0)
        self._memlock_ = Lock()

    def set_image_shape_x(self, image_shape):
        self.image_shape_x = image_shape
        self.cache_shape_x = (self.cache_size,) + self.image_shape_x

    def set_number_channels_x(self, num_channels):
        self.numE_x = num_channels

    def set_image_shape_y(self, image_shape):
        self.image_shape_y = image_shape
        self.cache_shape_y = (self.cache_size,) + self.image_shape_y

    def set_number_channels_y(self, num_channels):
        self.numE_y = num_channels

    def allocate(self):
        nitems_x = 1
        nitems_y = 1
        for i in range(0, len(self.cache_shape_x)):
            nitems_x *= self.cache_shape_x[i]
        for i in range(0, len(self.cache_shape_y)):
            nitems_y *= self.cache_shape_y[i]

        self.cacheX_base = Array(ctypes.c_float, nitems_x)
        self.cacheX = numpy.ctypeslib.as_array(self.cacheX_base.get_obj())
        self.cacheX = self.cacheX.reshape(self.cache_shape_x)
        self.limitsX_base = Array(ctypes.c_double, self.cache_size*self.numE_x*2)
        self.limitsX = numpy.ctypeslib.as_array(self.limitsX_base.get_obj())
        self.limitsX = self.limitsX.reshape((self.cache_size,self.numE_x,2))
        self.cacheY_base = Array(ctypes.c_float, nitems_y)
        self.cacheY = numpy.ctypeslib.as_array(self.cacheY_base.get_obj())
        self.cacheY = self.cacheY.reshape(self.cache_shape_y)
        self.limitsY_base = Array(ctypes.c_double, self.cache_size*self.numE_y*2)
        self.limitsY = numpy.ctypeslib.as_array(self.limitsY_base.get_obj())
        self.limitsY = self.limitsY.reshape((self.cache_size,self.numE_y,2))
        self.renew_cache = Value(ctypes.c_bool, True)
        self.cache_used_counter = Value('i', 0)
        self.cache_used_counter_x = Value('i', 0)
        self.cache_used_counter_y = Value('i', 0)
        self.cache_renewed_counter = Value('i', 0)

    def is_cache_updated(self):
        return (self.renew_cache.value is False)

    def get_number_renewed_items(self):
        return int(self.cache_renewed_counter.value)

    def get_renew_index(self):
        #self._memlock_.acquire()
        result = 0
        with self.cache_renewed_counter.get_lock():
            result = self.cache_renewed_counter.value
            self.cache_renewed_counter.value+=1
        if self.cache_renewed_counter.value >= self.cache_size:
            with self.renew_cache.get_lock():
                self.renew_cache.value = False
            with self.cache_renewed_counter.get_lock():
                self.cache_renewed_counter.value = 0
        if self.cache_used_counter.value >= self.cache_period:
            with self.cache_used_counter.get_lock():
                self.cache_used_counter.value = 0
            with self.cache_used_counter_x.get_lock():
                self.cache_used_counter_x.value = 0
            with self.cache_used_counter_y.get_lock():
                self.cache_used_counter_y.value = 0
        #self._memlock_.release()
        return result

    def get_cache_size(self):
        return self.cache_size

    def set_cache_item_x(self, index, item):
        #self._memlock_.acquire()
        #with self._memlock_:
        with self.cacheX_base.get_lock():
            self.cacheX[index]=item
        #print("Cache X - renewed items: {}; used items: {}". format(self.cache_renewed_counter, self.cache_used_counter_x))
        #self._memlock_.release()

    def set_item_limits_x(self, index, minval, maxval):
        with self.limitsX_base.get_lock():
            self.limitsX[index,:,0] = minval
            self.limitsX[index,:,1] = maxval

    def set_cache_item_y(self, index, item):
        #self._memlock_.acquire()
        #with self._memlock_:
        with self.cacheY_base.get_lock():
            self.cacheY[index]=item
        #self._memlock_.release()

    def set_item_limits_y(self, index, minval, maxval):
        with self.limitsY_base.get_lock():
            self.limitsY[index,:,0] = minval
            self.limitsY[index,:,1] = maxval

    def get_cache_item_x(self, index):
        result = None
        #self._memlock_.acquire()
        #with self._memlock_:
        with self.cacheX_base.get_lock():
            result = self.cacheX[index]
        with self.cache_used_counter_x.get_lock():
            self.cache_used_counter_x.value += 1
        with self.cache_used_counter.get_lock():
            self.cache_used_counter.value = max(self.cache_used_counter_x.value, self.cache_used_counter_y.value)
        if self.cache_used_counter.value >= self.cache_period:
            with self.renew_cache.get_lock():
                self.renew_cache.value = True
        #print("Cache X - renewed items: {}; used items: {}".format(self.cache_renewed_counter, self.cache_used_counter_x))
        #self._memlock_.release()
        return result

    def get_item_limits_x(self, index):
        minval = None
        maxval = None
        with self.limitsX_base.get_lock():
            minval = self.limitsX[index,:,0]
            maxval = self.limitsX[index,:,1]
        return numpy.squeeze(minval), numpy.squeeze(maxval)

    def get_cache_item_y(self, index):
        result = None
        #self._memlock_.acquire()
        #with self._memlock_:
        with self.cacheY_base.get_lock():
            result = self.cacheY[index]
        with self.cache_used_counter_y.get_lock():
            self.cache_used_counter_y.value += 1
        with self.cache_used_counter.get_lock():
            self.cache_used_counter.value = max(self.cache_used_counter_x.value, self.cache_used_counter_y.value)
        if self.cache_used_counter.value >= self.cache_period:
            with self.renew_cache.get_lock():
                self.renew_cache.value = True
        #self._memlock_.release()
        return result

    def get_item_limits_y(self, index):
        minval = None
        maxval = None
        with self.limitsY_base.get_lock():
            minval = self.limitsY[index,:,0]
            maxval = self.limitsY[index,:,1]
            return numpy.squeeze(minval), numpy.squeeze(maxval)

