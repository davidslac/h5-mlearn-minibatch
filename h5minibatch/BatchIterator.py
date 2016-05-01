from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import h5py

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
        
        self.TIMEOUT = 120
        h5 = h5py.File(self.samples.h5files[0], 'r')
        
        self.batchDatasets = {}
        for ds in datasets:
            dsDtype = h5[ds].dtype
            dsShape = h5[ds].shape
            assert len(dsShape)>1, "BatchIterator expects at least 2D datasets for batch reading"
            batchShape = tuple([self.batchsize] + list(dsShape[1:]))
            self.batchDatasets[ds] = np.zeros(batchShape, dtype=dsDtype)
        h5.close()
        del h5

    def __iter__(self):
        return self

    def next(self):
        self._readNextBatch()
        if self.totalEpochs and self.curEpoch >= self.totalEpochs:
            raise StopIteration()
        if self.batchLimit and self.batchLimit <= self.nextSampleIdx//self.batchsize:
            raise StopIteration()

        batchDict = {'epoch':self.curEpoch,
                     'batch':self.nextSampleIdx//self.batchsize,
                     'fvecs':self.samples.allFeatureVector[self.nextSampleIdx:(self.nextSampleIdx+self.batchsize)],
                     'labels':self.samples.allLabels[self.nextSampleIdx:(self.nextSampleIdx+self.batchsize)],
                     'filesRows':self.samples.allSamples[self.nextSampleIdx:(self.nextSampleIdx+self.batchsize)],
                     'readtime':self._batchReadTime,
                     'datasets':{},
        }
        for nm in self.datasets:
            batchDict['datasets'][nm] = self.batchDatasets[nm].copy()

        if self.nextSampleIdx + self.batchsize >= self.samples.totalSamples:
            self.curEpoch += 1
            self.nextSampleIdx = 0
            self.samples.shuffle()
        else:
            self.nextSampleIdx += self.batchsize
        return batchDict

    def _readNextBatch(self):
        t0 = time.time()

        startIdx = self.nextSampleIdx
        endIdx = startIdx + self.batchsize
        
        fileRow = self.samples.allSamples[startIdx:endIdx]
        fileRowBatchRow = np.zeros(self.batchsize, dtype=([('file',np.int64),
                                                            ('row',np.int64),
                                                            ('batchrow',np.int64)]))
        fileRowBatchRow['file']=fileRow['file']
        fileRowBatchRow['row']=fileRow['row']
        fileRowBatchRow['batchrow']=np.arange(self.batchsize)

        sortedFileRowBatchRow = np.sort(fileRowBatchRow)

        lastFileIdx = -1
        h5 = None
        for frb in sortedFileRowBatchRow:
            fileIdx, fileRow, batchRow = frb
            if lastFileIdx != fileIdx:
                if h5: h5.close()
                h5 = h5py.File(self.samples.h5files[fileIdx],'r')
                lastFileIdx = fileIdx
            for ds in self.datasets:
                self.batchDatasets[ds][batchRow] = h5[ds][fileRow]
        h5.close()
        self._batchReadTime = time.time()-t0

    def fvecStats(self, verbose=False, h5save=None, force=False):
        return self.samples.fvecStats(verbose, h5save, force)
