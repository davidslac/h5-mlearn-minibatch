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
                 batchsize,
                 num_batches,
                 h5batch_reader):

        assert batchsize <= samples.totalSamples, "batchsize=%d to big for number of samples=%d" % (batchsize, samples.totalSamples)
        self.totalEpochs=epochs
        self.batchsize=batchsize
        self.num_batches = num_batches
        self.samples=samples
        self.h5br = h5batch_reader

        self.curEpoch=0
        self.curBatch=0
        self.nextSampleIdx=0

        self.batchesPerEpoch = self.samples.totalSamples//self.batchsize
        
    def __iter__(self):
        return self

    def next(self):
        batchDict = self._readNextBatch()
        if self.totalEpochs>0 and self.curEpoch >= self.totalEpochs:
            raise StopIteration()
        if self.num_batches>0 and self.curBatch >= self.num_batches:
            raise StopIteration()
        self.nextSampleIdx += self.batchsize
        self.curBatch += 1
        if self.nextSampleIdx + self.batchsize > self.samples.totalSamples:
            self.curEpoch += 1
            self.nextSampleIdx = 0
            self.samples.shuffle()
        return batchDict

    def _readNextBatch(self):
        t0 = time.time()

        startIdx = self.nextSampleIdx
        endIdx = startIdx + self.batchsize
        assert endIdx <= self.samples.totalSamples
        
        fileRow = self.samples.allSamples[startIdx:endIdx]

        batchDict = {'epoch':self.curEpoch,
                     'batch':self.curBatch % self.batchesPerEpoch,
                     'filesRows':fileRow.copy(),
                     'readtime':None,
                     'size':self.batchsize,
                     'dsets':{},
                     'dset_groups':{},
        }
        
        fileRowBatchRow = np.empty(self.batchsize, dtype=([('file',np.int64),
                                                           ('row',np.int64),
                                                           ('batchrow',np.int64)]))
        fileRowBatchRow['file']=fileRow['file'].copy()
        fileRowBatchRow['row']=fileRow['row'].copy()
        fileRowBatchRow['batchrow']=np.arange(self.batchsize)

        sortedFileRowBatchRow = np.sort(fileRowBatchRow)

        lastFileIdx = -1
        h5 = None
        for frb in sortedFileRowBatchRow:
            fileIdx, fileRow, batchRow = frb
            if lastFileIdx != fileIdx:
                if h5: h5.close()
                h5 = h5py.File(self.samples.h5br.h5files[fileIdx],'r')
                lastFileIdx = fileIdx
            for ds in self.h5br.dsets:
                val = h5[ds][fileRow]
                if ds not in batchDict['dsets']:
                    shape = tuple([self.batchsize] + list(val.shape))
                    batchDict['dsets'][ds] = np.empty(shape=shape, dtype=h5[ds].dtype)
                batchDict['dsets'][ds][batchRow] = val
                    
            for ds_group in self.h5br.dset_groups:
                val = ds_group.read_sample(h5, fileRow)
                if ds_group.name not in batchDict['dset_groups']:
                    shape = tuple([self.batchsize] + list(val.shape))
                    batchDict['dset_groups'][ds_group.name] = np.empty(shape=shape, dtype=ds_group.dtype)
                batchDict['dset_groups'][ds_group.name][batchRow] = val

        h5.close()
        batchDict['readtime'] = time.time()-t0
        return batchDict
