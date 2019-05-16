
"""
Modified keras callbacks

Author: Soeren Gregersen, 2018
"""

from __future__ import print_function

import os
import re
import keras
import math


class ModelCheckpoint(keras.callbacks.Callback):
    """Save the model after every epoch, now extended with auto-remove.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        modelpath: string, path to save the model file.
        weightspath: string or `None`, path to save the weights file. If None,
            the weightspath is generated from modelpath.
            `weightspath = filename_without_ext(modelpath) + '_weights' + ext`
        verbose: verbosity mode, 0 or 1.
        save_model: `True` or `False`, if `True` will save model.
        save_weights: `True` or `False`, if `True` will save weights.
        period: number, interval (number of epochs) between checkpoints.
        auto_remove_model: `True` or `False`, if `True` will remove previously
            saved model files. Files can be kept (skipped for removal) if
            keep_period != 0.
        auto_remove_weight: `True` or `False`, same as auto_remove_model, but
            for weights instead.
        keep_period: number, interval (number of epochs) between kept
            checkpoints (see auto_remove). Default is infinit i.e. never keep.
    """

    def __init__(self, modelpath, weightspath=None,
                 verbose=0, save_model=True, save_weights=True, period=1,
                 auto_remove_model=False, auto_remove_weights=False,
                 keep_period=math.inf):
        super(ModelCheckpoint, self).__init__()
        self.verbose = verbose
        self.modelpath = modelpath
        self.weightspath = weightspath
        if weightspath is None:
            p = "{}_weights{}".format(*os.path.splitext(weightspath))
            self.weightspath = p
        self.save_model = save_model
        self.save_weights = save_weights
        self.period = period
        self.auto_remove_model = auto_remove_model
        self.auto_remove_weights = auto_remove_weights
        self.keep_period = keep_period

        self.epochs_since_last_save = 0
        self.epochs_since_last_remove = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        self.epochs_since_last_remove += 1

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            txt = 'Epoch {:05d}'.format(epoch) + ': saving {} to {}'
            modelpath = self.modelpath.format(epoch=epoch + 1, **logs)
            weightspath = self.weightspath.format(epoch=epoch + 1, **logs)
            if self.save_model:
                if self.verbose > 0:
                    print(txt.format('model', modelpath))
                self.model.save(modelpath, overwrite=True)
            if self.save_weights:
                if self.verbose > 0:
                    print(txt.format('weights', weightspath))
                self.model.save_weights(weightspath, overwrite=True)

        if self.epochs_since_last_remove <= self.keep_period:
            txt = 'auto-removing {} file {}'
            modelpath = self.modelpath.format(epoch=epoch, **logs)
            weightspath = self.weightspath.format(epoch=epoch, **logs)
            if self.auto_remove_model and os.path.exists(modelpath):
                if self.verbose > 0:
                    print(txt.format('model', modelpath))
                os.remove(modelpath)
            if self.auto_remove_weights and os.path.exists(weightspath):
                if self.verbose > 0:
                    print(txt.format('weights', weightspath))
                os.remove(weightspath)
        else:
            self.epochs_since_last_remove = 0

    @staticmethod
    def last_checkpoint_epoch_and_model(modelpath):

        def get_format_args(string, pattern):
            regex = re.sub(r'{([^:}]*?)(?::[^}]*)?}', r'(?P<_\1>.+)', pattern)
            keys = re.findall(r'{([^:}]*?)(?::[^}]*)?}', pattern)
            result = re.fullmatch(regex, string)
            if result is None:
                return {k: None for k in keys}
            values = list(result.groups())
            _dict = dict(zip(keys, values))
            return _dict

        latest_epoch = 0
        latest_model = None
        dirpath = os.path.dirname(modelpath)
        for file in os.listdir(dirpath):
            file = os.path.join(dirpath, file)
            format_args = get_format_args(file, modelpath)

            if 'epoch' in format_args and format_args['epoch'] is not None:
                if int(format_args['epoch']) > latest_epoch:
                    latest_epoch = int(format_args['epoch'])
                    latest_model = file
        return latest_epoch, latest_model

    @staticmethod
    def remove_all_checkpoints(modelpath, weightspath=None):
        if weightspath is None:
            weightspath = "{}_weights{}".format(*os.path.splitext(weightspath))

        def matches_format_pattern(string, pattern):
            regex = re.sub(r'{([^:}]*?)(?::[^}]*)?}', r'(?P<_\1>.+)', pattern)
            return re.fullmatch(regex, string) is not None

        for path in [modelpath, weightspath]:
            dirpath = os.path.dirname(path)
            for file in os.listdir(dirpath):
                file = os.path.join(dirpath, file)
                if matches_format_pattern(file, path):
                    os.remove(file)
