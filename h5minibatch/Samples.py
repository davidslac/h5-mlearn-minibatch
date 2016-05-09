from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import copy
import os
import time
import warnings
import random

import numpy as np
import h5py

from .utils import makeMask

class Samples(object):
    def __init__(self, 
                 h5files, 
                 label_dataset, 
                 fvec_datasets,
                 fvec_dtype,
                 include_if_one_mask_datasets=[], 
                 exclude_if_negone_mask_datasets=[],
                 verbose=False):

        assert len(h5files)
        assert len(h5files)==len(set(h5files)), "h5files are not distinct"
        self.h5files = copy.deepcopy(h5files)
        self.label_dataset=label_dataset
        self.fvec_datasets=fvec_datasets
        self.fvec_dtype=fvec_dtype
        self.include_if_one_mask_datasets=include_if_one_mask_datasets
        self.exclude_if_negone_mask_datasets=exclude_if_negone_mask_datasets
        self.verbose=verbose

        if verbose:
            t0 = time.time()
            print("Samples: scanning through files %d h5files, saving %d datasets into fvec..." % 
                  (len(h5files), len(fvec_datasets)))
            sys.stdout.flush()
            
        sampleDtype = np.dtype([('file', np.int64),
                                ('row', np.int64)])
        allSamples = np.zeros(0, dtype=sampleDtype)
        allLabels = np.zeros(0, np.int64)
        allFeatureVector = np.zeros((0,len(fvec_datasets)), 
                                     dtype = fvec_dtype)

        totalSamples = 0
        for h5Idx, h5filename in enumerate(self.h5files):
            h5 = h5py.File(h5filename, 'r')
            assert label_dataset in h5.keys(), "label datset=%s not in h5file=%s" % (label_dataset, h5filename)
            assert np.issubdtype(h5[label_dataset].dtype, np.integer), "label dataset=%s is not integeral, it is %s" % \
                (label_dataset, h5[label_dataset].dtype)
            mask = makeMask(h5, 
                            exclude_if_negone_mask_datasets = exclude_if_negone_mask_datasets,
                            include_if_one_mask_datasets = include_if_one_mask_datasets)
            numSamplesThisFile = np.sum(mask)
            totalSamples += numSamplesThisFile
            samplesThisFile = np.zeros(numSamplesThisFile,  sampleDtype)
            samplesThisFile['file'] = h5Idx
            rowsThisFile = np.arange(len(h5[label_dataset]))[mask]
            samplesThisFile['row'] = rowsThisFile
            labelsThisFile = h5[label_dataset][mask]
            fvecThisFile = np.zeros((numSamplesThisFile, len(fvec_datasets)),
                                    dtype=fvec_dtype)
            for nmIdx,nm in enumerate(fvec_datasets):
                assert nm in h5.keys(), "dataset=%s for feature vector not in h5file=%s" % (nm, h5filename)
                fvecThisFile[:,nmIdx] = h5[nm][mask]
            allSamples = np.concatenate((allSamples, samplesThisFile))
            allLabels = np.concatenate((allLabels, labelsThisFile))
            allFeatureVector = np.concatenate((allFeatureVector, fvecThisFile))
        
        assert allSamples.shape == (totalSamples,)
        assert allLabels.shape == (totalSamples,)
        assert allFeatureVector.shape == (totalSamples, len(fvec_datasets))

        self.allSamples = allSamples
        self.allLabels = allLabels
        self.allFeatureVector = allFeatureVector
        self.totalSamples = totalSamples

        if verbose:
            sys.stdout.write("... scan took %.2f sec. There are %d samples.\n" % 
                             (time.time()-t0, self.totalSamples))
            sys.stdout.flush()

    def numOutputs(self):
        return np.max(self.allLabels)

    def shuffle(self):
        perm=np.arange(self.totalSamples)
        np.random.shuffle(perm)
        self.allSamples = self.allSamples[perm]
        self.allLabels = self.allLabels[perm]
        self.allFeatureVector = self.allFeatureVector[perm]

    def split(self, fractions, batchsize):
        assert np.sum(np.array(fractions)) <=1.0
        subSampleList = []
        lastIdx = 0
        while len(fractions)>0:
            nextIdx = lastIdx + int(round(fractions.pop(0)*self.totalSamples))
            nextIdx = min(nextIdx, self.totalSamples)
            if nextIdx - lastIdx < batchsize:
                if lastIdx + batchsize <= self.totalSamples:
                    nextIdx = lastIdx + batchsize
                else:
                    print("split: warning - empty partition")
                    nextIdx = lastIdx
            else:
                nextIdx -= (nextIdx % batchsize)
            subSampleList.append(SubSamples(h5files=self.h5files,
                                            allSamples=self.allSamples[lastIdx:nextIdx],
                                            labels=self.allLabels[lastIdx:nextIdx],
                                            fvec_datasets=self.fvec_datasets,
                                            fvec=self.allFeatureVector[lastIdx:nextIdx],
                                            verbose=self.verbose))
            lastIdx = nextIdx

        lastGroupSize = self.totalSamples - lastIdx
        discard = lastGroupSize % batchsize
        if self.verbose:
            print("Samples.split - discarding %d samples to make each split divisible by %d" %
                  (discard, batchsize))
        nextIdx = self.totalSamples - discard
        subSampleList.append(SubSamples(h5files=self.h5files,
                                        allSamples=self.allSamples[lastIdx:nextIdx],
                                        labels=self.allLabels[lastIdx:nextIdx],
                                        fvec_datasets=self.fvec_datasets,
                                        fvec=self.allFeatureVector[lastIdx:nextIdx],
                                        verbose=self.verbose))
        return subSampleList
        
    def fvecStats(self, verbose=False, h5save=None, force=False):
        nmWidth = max([len(nm) for nm in self.fvec_datasets])
        stats = {}
        for idx,nm in enumerate(self.fvec_datasets):
            mean=np.mean(self.allFeatureVector[:,idx])
            std=np.std(self.allFeatureVector[:,idx])
            maxval=np.max(self.allFeatureVector[:,idx])
            minval=np.min(self.allFeatureVector[:,idx])
            maxstd = (maxval-mean)/std
            minstd = (mean-minval)/std
            mxdevstd = max(maxstd, minstd)
            stats[nm]={'mean':mean, 'std':std, 'max':maxval, 'minval':minval, 
                       'mxdevstd':mxdevstd}
            if verbose:
                print("#  %s: mean=%20.5f std=%15.5f mx=%16.1f mn=%16.1f mxdevstd=%10.1f" % 
                      (nm.rjust(nmWidth), mean, std, maxval, minval, mxdevstd))
        if h5save:
            assert force or not os.path.exists(h5save), "fvecStats -- set force=True to overwrite %s" % h5save
            h5 = h5py.File(h5save, 'w')
            for nm, nmDict in stats.iteritems():
                for st, val in nmDict.iteritems():
                    h5['%s:%s' % (nm, st)] = val
            h5.close()
        return stats
        
class SubSamples(Samples):
    def __init__(self, h5files, allSamples, labels, fvec_datasets, fvec, verbose):
        self.h5files = copy.deepcopy(h5files)
        self.allSamples = allSamples.copy()
        self.allLabels = labels.copy()
        self.fvec_datasets = fvec_datasets
        self.allFeatureVector = fvec.copy()
        self.verbose=verbose
        self.totalSamples = len(self.allSamples)
        if self.verbose:
            print("SubSamples with %d samples" % self.totalSamples)
            sys.stdout.flush()

