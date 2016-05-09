from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import copy
import time
import warnings

import numpy as np
import h5py

from .utils import makeMask, str2np_dtype
from .Samples import Samples
from .BatchIterator import BatchIterator
from .FvecPreprocess import loadFvecStats

class H5MiniBatchReader(object):
    '''delivers minibatches of feature/labels from a collection of h5 files
    '''
    def __init__(self, 
                 h5files,
                 include_if_one_mask_datasets = [],
                 exclude_if_negone_mask_datasets = [],
                 verbose = False):
        '''initialize with set of h5files, and names of datasets
        used to identify which samples to include in iterations
       like getAll.
        '''
        _h5files = copy.deepcopy(h5files)
        original_len_h5files = len(_h5files)
        _h5files = list(set(_h5files))
        if len(_h5files) != original_len_h5files:
            warnings.warn("H5MiniBatchReader: duplicate filenames "
                          "given in h5files argument - uniqifying")
        _h5files.sort()

        self._h5files = _h5files
        self._include_if_one_mask_datasets = include_if_one_mask_datasets
        self._exclude_if_negone_mask_datasets = exclude_if_negone_mask_datasets
        self._verbose = verbose

    def h5keys(self):
        '''keys in the h5 files'''
        return h5py.File(self._h5files[0],'r').keys()

    def initClassify(self,
                     label_dataset,
                     datasets_for_feature_vector=[],
                     feature_image_datasets=[],
                     fvec_whitten=False,
                     fvec_whitten_fname=None,
                     fvec_dtype = 'float32',
                     minibatch_size=64,
                     validation_size_fraction=0.05,                
                     test_size_fraction=0.05,
                     random_seed=None):

        self._label_dataset = label_dataset
        self._datasets_for_feature_vector = datasets_for_feature_vector
        self._feature_image_datasets = feature_image_datasets
        self._fvec_whitten = fvec_whitten
        self._fvec_whitten_fname = fvec_whitten_fname
        self._fvec_dtype = str2np_dtype(fvec_dtype)
        self._minibatch_size = minibatch_size
        self._validation_size_fraction = validation_size_fraction
        self._test_size_fraction = test_size_fraction

        self._fvec_whitten_stats = None

        if random_seed is not None:
            np.random.seed(random_seed)
            if self._verbose:
                print("H5MiniBatchReader: set random_seed to %r" % random_seed)

        samples = Samples(h5files=self._h5files,
                          label_dataset=self._label_dataset,
                          fvec_datasets=self._datasets_for_feature_vector,
                          fvec_dtype=self._fvec_dtype,
                          include_if_one_mask_datasets=self._include_if_one_mask_datasets,
                          exclude_if_negone_mask_datasets=self._exclude_if_negone_mask_datasets,
                          verbose=self._verbose)

        samples.shuffle()
        sampleList = samples.split([test_size_fraction, validation_size_fraction], self._minibatch_size)
        self.samples = {}
        self.samples['test'], self.samples['validation'], self.samples['train'] = sampleList

        if self._fvec_whitten:
            if not os.path.exists(self._fvec_whitten_fname):
                self.samples['train'].fvecStats(self._verbose, 
                                                self._fvec_whitten_fname)
            self._fvec_whitten_stats = loadFvecStats(self._fvec_whitten_fname)
                                                            
    def image_dataset_shapes(self):
        shapes = {}
        for nm in self._feature_image_datasets:
            shapes[nm] = h5py.File(self._h5files[0], 'r')[nm].shape[1:]
        return shapes

    def datasets_for_feature_vector(self):
        return self._datasets_for_feature_vector

    def numSamples(self, partition):
        return self.samples[partition].totalSamples

    def numOutputs(self):
        return 1+max([self.samples[partition].numOutputs() for partition in 
                      ['test','validation','train']])

    def numBatchesPerEpoch(self, partition):
        return self.samples[partition].totalSamples // self._minibatch_size

    def batchIterator(self, partition, epochs=None, batchLimit=None):
        return BatchIterator(samples=self.samples[partition],
                             epochs=epochs,
                             batchLimit=batchLimit,
                             batchsize = self._minibatch_size,
                             datasets = self._feature_image_datasets,
                             fvec_whitten = self._fvec_whitten,
                             fvec_whitten_stats = self._fvec_whitten_stats)

