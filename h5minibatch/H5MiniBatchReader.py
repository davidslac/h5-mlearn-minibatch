from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import copy
import time
import warnings

import numpy as np
import h5py

from .utils import makeMask
from .Samples import Samples
from .BatchIterator import BatchIterator

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

    def prepareClassification(self,
                              label_dataset,
                              datasets_for_feature_vector=[],
                              feature_vector_dtype = np.float32,
                              feature_image_datasets=[],
                              minibatch_size=64,
                              validation_size_percent=0.05,                
                              test_size_percent=0.05,
                              random_seed=None):

        self._label_dataset = label_dataset
        self._datasets_for_feature_vector = datasets_for_feature_vector
        self._feature_image_datasets = feature_image_datasets
        self._minibatch_size = minibatch_size
        self._validation_size_percent = validation_size_percent
        self._test_size_percent = test_size_percent

        if random_seed is not None:
            np.random.seed(random_seed)
            if self._verbose:
                print("H5MiniBatchReader: set random_seed to %r" % random_seed)

        samples = Samples(h5files=self._h5files,
                          label_dataset=self._label_dataset,
                          fvec_datasets=self._datasets_for_feature_vector,
                          fvec_dtype=feature_vector_dtype,
                          include_if_one_mask_datasets=self._include_if_one_mask_datasets,
                          exclude_if_negone_mask_datasets=self._exclude_if_negone_mask_datasets,
                          verbose=self._verbose)

        samples.shuffle()
        sampleList = samples.split([validation_size_percent, test_size_percent], self._minibatch_size)
        self.samples = {}
        self.samples['test'], self.samples['validation'], self.samples['train'] = sampleList

    def numSamples(self, partition):
        return self.samples[partition].totalSamples
            
    def numBatchesPerEpoch(self, partition):
        return self.samples[partition].totalSamples // self._minibatch_size
        

    def batchIterator(self, partition, epochs, batchLimit=None):
        return BatchIterator(samples=self.samples[partition],
                             epochs=epochs,
                             batchLimit=batchLimit,
                             batchsize = self._minibatch_size,
                             datasets = self._feature_image_datasets)

