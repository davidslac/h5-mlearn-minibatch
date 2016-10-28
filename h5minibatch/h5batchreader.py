from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import copy
import time

import numpy as np
import h5py

#from .utils import makeMask, str2np_dtype
from .batchiterator import BatchIterator
from .datasetgroup import DataSetGroup
from .samples import Samples

class H5BatchReader(object):
    '''delivers batches of numpy arrays from a collection of h5 files.
    Each h5 file should have the same schema - just splits one big dataset into chunks.
    Works with datasets from the root of the h5 files.
    The datasets need to be aligned arrays. The same row from different
    datasets corresponds to the same sample for a batch.
    some datasets can be merged into a feature vector.
    other datasets can be used to mask out rows (bad samples).

    balance - you can balance the samples based on a dataset.
    '''
    def __init__(self, 
                 h5files,
                 dsets = [],
                 dset_groups = [],
                 include_if_one_mask_datasets = [],
                 exclude_if_negone_mask_datasets = [],
                 verbose = False):

        self.logname = 'H5BatchReader'

        assert isinstance(h5files, list)
        assert len(h5files)>0
        assert isinstance(dsets, list)
        assert isinstance(dset_groups, list)
        assert len(dsets)+len(dset_groups)>0
        
        h5files = copy.deepcopy(h5files)
        original_len_h5files = len(h5files)
        h5files = list(set(h5files))
        if len(h5files) != original_len_h5files:
            warn("%s: duplicate filenames given in h5files argument - uniqifying" % self.logname)
        h5files.sort()

        self.h5files = h5files
        self.include_if_one_mask_datasets = include_if_one_mask_datasets
        self.exclude_if_negone_mask_datasets = exclude_if_negone_mask_datasets
        self.verbose = verbose

        self.dsets = copy.deepcopy(dsets)
        self.dset_groups = copy.deepcopy(dset_groups)

        self.all_samples = Samples(self)
        self.train_samples = self.all_samples
        self.validation_samples = None
        self.test_samples = None
        
    def any_dset(self):
        dset = None
        if len(self.dsets)>0:
            dset = self.dsets[0]
        elif len(self.dset_groups)>0:
            dset = self.dset_groups[0].dsets[0]
        assert dset, "no dsets or dset_groups."
        return dset
    
    def num_rows_this_file(self, fname):
        dset = self.any_dset()
        h5 = h5py.File(fname,'r')
        assert dset in h5
        return len(h5[dset])            
            
    def split(self, train=80, validation=10, test=10):
        assert train+validation+test==100, "split: train+validate+test must add up to 100, can set some to 0"
        subSampleList = self.all_samples.split(train, validation, test)
        self.train_samples, self.validation_samples, self.test_samples = subSampleList

    def h5keys(self):
        '''keys in the h5 files'''
        return h5py.File(self.h5files[0],'r').keys()

    def get_shapes(self):
        '''returns X,Y
        X is a list of shapes for the features,
        Y is a list of shapes for the labels
        shape for a scalar is None
        '''
        h5 = h5py.File(self.h5files[0],'r')
        X = []
        Y = []
        
        for L,feats in zip([X,Y],
                           [self.features, self.labels]):
            for feat in feats:
                if isinstance(feat, FeatureVectorFromH5_Datasets):
                    L.append(feat.sample_shape_from_h5(self.h5files[0]))
                else:
                    name = feat
                    dset = h5[name]
                    shape = dset.shape
                    if len(shape)==1:
                        L.append(None)
                    else:
                        L.append(shape[1:])
        return X,Y

    def get_label_dtypes(self):
        h5 = h5py.File(self.h5files[0],'r')
        Y = []
        
        for name in self.labels:
            dset = h5[name]
            Y.append(dset.dtype)
        return Y

    def numSamples(self, partition):
        return self.samples[partition].totalSamples

    def numOutputs(self):
        return 1+max([self.samples[partition].numOutputs() for partition in 
                      ['test','validation','train']])

    def numBatchesPerEpoch(self, partition):
        return self.samples[partition].totalSamples // self.minibatch_size

    def train_iter(self, batchsize, epochs=0, num_batches=0):
        if not self.train_samples:
            return None
        return BatchIterator(samples=self.train_samples,
                             batchsize = batchsize,
                             epochs = epochs,
                             num_batches = num_batches,
                             h5batch_reader = self)

    def validation_iter(self, batchsize, epochs=0, num_batches=0):
        if not self.validation_samples:
            return None
        return BatchIterator(samples=self.validation_samples,
                             batchsize = batchsize,
                             epochs = epochs,
                             num_batches = num_batches,
                             h5batch_reader = self)

    def test_iter(self, batchsize, epochs=0, num_batches=0):
        if not self.test_samples:
            return None
        return BatchIterator(samples=self.test_samples,
                             batchsize = batchsize,
                             epochs = epochs,
                             num_batches = num_batches,
                             h5batch_reader = self)

