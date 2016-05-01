from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class BatchIterator(object):
    def __init__(self,
                 samples,
                 epochs,
                 batchLimit,
                 batchsize,
                 datasets):
        
        assert samples.totalSamples % batchsize == 0
        self.totalEpochs=epochs
        self.batchLimit=batchLimit
        self.batchsize=batchsize
        self.samples=samples
        self.datasets=datasets

        self.curEpoch=0
        self.nextSampleIdx=0
        self.batchesPerEpoch = self.samples.totalSamples//self.batchsize

        self._readNextBatch()

    def __iter__(self):
        return self

    def next(self):
        self._waitForNextBatchToBeRead()
        if self.totalEpochs and self.curEpoch >= self.totalEpochs:
            raise StopIteration()
        if self.batchLimit and self.batchLimit <= self.nextSampleIdx//self.batchsize:
            raise StopIteration()

        batchDict = {'epoch':self.curEpoch,
                     'batch':self.nextSampleIdx//self.batchsize,
                     'fvecs':self.samples.allFeatureVector[self.nextSampleIdx:(self.nextSampleIdx+self.batchsize)],
                     'labels':self.samples.allLabels[self.nextSampleIdx:(self.nextSampleIdx+self.batchsize)],
                     'datasets':{},
        }
        for nm in self.datasets:
            batchDict['datasets'][nm] = self._nextBatchDataset(nm)

        if self.nextSampleIdx + self.batchsize >= self.samples.totalSamples:
            self.curEpoch += 1
            self.nextSampleIdx = 0
            self.samples.shuffle()
        else:
            self.nextSampleIdx += self.batchsize

        self._readNextBatch()

        return batchDict

    def _readNextBatch(self):
        '''release GIL, run background thread to to 
        read all the datsets
        '''
        startIdx = self.nextSampleIdx
        endIdx = startIdx + self.batchsize
        
        filesRows = self.samples.allSamples[startIdx:endIdx]
        
        pass

    def _waitForNextBatchToBeRead(self):
        '''wait for above
        '''
        pass

    def _nextBatchDataset(self, nm):
        '''after wait, now safe to get arrays and use them
        '''
        return np.zeros((2,2))
