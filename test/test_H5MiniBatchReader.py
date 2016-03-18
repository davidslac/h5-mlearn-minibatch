from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest
import uuid
import numpy as np
import h5py

from h5minibatch.H5MiniBatchReader import H5MiniBatchReader

class TestRaw( unittest.TestCase ) :
    def setUp(self):
        self.h5filename = 'unitest_' + uuid.uuid4().hex + '.h5'
        h5 = h5py.File(self.h5filename,'w')
        img_features = np.zeros((100,2,3),np.int16)
        img_features += 10
        img_labels = np.zeros(100, np.int32)
        h5['features']=img_features
        h5['labels']=img_labels
        h5.close()

    def tearDown(self):
        os.unlink(self.h5filename)

    def testHaveRaw(self):
        miniBatchReader = H5MiniBatchReader(h5files=[self.h5filename],
                                            minibatch_size=2,
                                            validation_size=2,
                                            feature_preprocess=None,
                                            verbose=True)
        train_feats, train_labels = miniBatchReader.get_all_train()
        validation_feats, validation_labels = miniBatchReader.get_validation_set()
        miniBatch_feats, miniBatch_labels = miniBatchReader.get_next_minibatch()
        assert train_feats.dtype == np.int16, "dtype is not int16, it is %r" % train_feats.dtype
        assert validation_feats.dtype == np.int16, "dtype is not int16, it is %r" % validation_feats.dtype
        assert miniBatch_feats.dtype == np.int16, "dtype is not int16, it is %r" % miniBatch_feats.dtype

    def testCacheRaw(self):
        miniBatchReader = H5MiniBatchReader(h5files=[self.h5filename],
                                            minibatch_size=2,
                                            validation_size=0,
                                            feature_preprocess=['mean'],
                                            max_mb_to_preload_all=100000,
                                            verbose=True)
        assert miniBatchReader._train_samples_cache._features.dtype == np.int16
        feats, labels = miniBatchReader.get_next_minibatch()
        assert feats.dtype == np.float32
        assert abs(feats[0,0,0])<.1, "preprocess has subtract mean, but feats[0,0,0]==%.2f" % feats[0,0,0]

    def testCacheProcessed(self):
        miniBatchReader = H5MiniBatchReader(h5files=[self.h5filename],
                                            minibatch_size=2,
                                            validation_size=0,
                                            feature_preprocess=['mean'],
                                            max_mb_to_preload_all=100000,
                                            cache_preprocess=True,
                                            verbose=True)
        assert miniBatchReader._train_samples_cache._features.dtype == np.float32
        feats, labels = miniBatchReader.get_next_minibatch()
        assert feats.dtype == np.float32
        assert feats.dtype == np.float32
        assert abs(feats[0,0,0])<.1, "preprocess has subtract mean, but feats[0,0,0]==%.2f" % feats[0,0,0]



class TestReadEpochs( unittest.TestCase ) :

    def setUp(self) :
        self.numH5Files = 3
        self.numFeatures = 100
        self.numOutputs = 8
        self.h5filenames = []
        featVal = 0
        for idx in range(self.numH5Files):
            h5filename = 'unitest_' + uuid.uuid4().hex + ('_%3.3d' % idx) + '.h5'
            h5 = h5py.File(h5filename,'w')
            img_features = np.zeros((self.numFeatures,2,3),np.float32)
            img_labels = np.zeros(self.numFeatures, np.int32)
            for ii in range(self.numFeatures):
                img_features[ii,:] = featVal
                img_labels[ii] = featVal % self.numOutputs
                featVal += 1
            h5['img'] = img_features
            h5['pk.label'] = img_labels
            h5.close()
            self.h5filenames.append(h5filename)
                
    def test_readThreeEpochs(self):
        numSamples = self.numFeatures * self.numH5Files
        sampleCounts = {}
        for ii in range(numSamples):
            sampleCounts[ii] = 0
        assert numSamples % 10 == 0
        numBatches = numSamples // 10
        miniBatchReader = H5MiniBatchReader(h5files=self.h5filenames,
                                            minibatch_size=10,
                                            validation_size=0,
                                            feature_dataset='img',
                                            label_dataset='pk.label',
                                            return_as_one_hot=True,
                                            feature_preprocess=None,
                                            number_of_batches=None,
                                            class_labels_max_imbalance_ratio=None,
                                            max_mb_to_preload_all=None,
                                            random_seed=None,
                                            verbose=True)

        assert all([x==y for x,y in zip(miniBatchReader.features_placeholder_shape(),
                                        [None, 2,3,1])])

        assert miniBatchReader.num_outputs()==self.numOutputs

        valid_feats, valid_labels = miniBatchReader.get_validation_set()
        assert valid_feats is None
        assert valid_labels is None

        epochs = 3
        numZeros = []
        for step_number in range(numBatches*epochs):
            features, labels = miniBatchReader.get_next_minibatch()
            numZerosThisMinibatch = 0
            for row in range(len(features)):
                featVal = int(features[row,:].flatten()[0])
                expectedLabel = featVal % self.numOutputs
                assert np.sum(labels[row,:])==1
                assert  1  == labels[row, expectedLabel]
                sampleCounts[featVal] += 1
                if expectedLabel == 0:
                    numZerosThisMinibatch += 1
            numZeros.append(numZerosThisMinibatch)
        assert np.std(numZeros)>0.1, "not enough randomness in the minibatches"
        for featVal, count in sampleCounts.iteritems():
            assert count == epochs

    def test_unbalanced(self):
        # unbalance the labels, make 50% 0
        for h5file in self.h5filenames:
            h5 = h5py.File(h5file,'r+')
            labels = h5['pk.label'][:]
            labels[0:len(labels)//2]=0
            h5['pk.label'][:]=labels[:]

        miniBatchReader = H5MiniBatchReader(h5files=self.h5filenames,
                                            minibatch_size=10,
                                            validation_size=24,
                                            feature_dataset='img',
                                            label_dataset='pk.label',
                                            return_as_one_hot=True,
                                            feature_preprocess=['log','mean'],
                                            number_of_batches=None,
                                            class_labels_max_imbalance_ratio=1.2,
                                            max_mb_to_preload_all=10,
                                            random_seed=None,
                                            verbose=True)
        train_feats, train_labels = miniBatchReader.get_all_train()
        validation_feats, validation_labels = miniBatchReader.get_validation_set()
        # the unbalancing leave 154 samples, validation is 24, leaving 130 
        # samples evenly divided by minibatch of 10 into 13 batches
        assert len(train_feats)==130, "len(train_feats)=%s" % (len(train_feats),)

        num_labels=np.sum(np.vstack((train_labels, validation_labels)),0)
        largest = max(num_labels)
        smallest = min(num_labels)
        assert 1.2 * smallest >= largest - 0.5, "1.2*smallest=%d -> %d bust largest=%d" % \
            (smallest, 1.2*smallest, largest)
        

    def test_unbalanced_no_cache(self):
        # unbalance the labels, make 50% 0
        for h5file in self.h5filenames:
            h5 = h5py.File(h5file,'r+')
            labels = h5['pk.label'][:]
            labels[0:len(labels)//2]=0
            h5['pk.label'][:]=labels[:]

        miniBatchReader = H5MiniBatchReader(h5files=self.h5filenames,
                                            minibatch_size=10,
                                            validation_size=24,
                                            feature_dataset='img',
                                            label_dataset='pk.label',
                                            return_as_one_hot=True,
                                            feature_preprocess=['log','mean'],
                                            number_of_batches=None,
                                            class_labels_max_imbalance_ratio=1.2,
                                            max_mb_to_preload_all=.001,
                                            random_seed=None,
                                            verbose=True)
        train_feats, train_labels = miniBatchReader.get_all_train()
        validation_feats, validation_labels = miniBatchReader.get_validation_set()
        # the unbalancing leave 154 samples, validation is 24, leaving 130 
        # samples evenly divided by minibatch of 10 into 13 batches
        assert len(train_feats)==130, "len(train_feats)=%s" % (len(train_feats),)

        num_labels=np.sum(np.vstack((train_labels, validation_labels)),0)
        largest = max(num_labels)
        smallest = min(num_labels)
        assert 1.2 * smallest >= largest - 0.5, "1.2*smallest=%d -> %d bust largest=%d" % \
            (smallest, 1.2*smallest, largest)
        

        labels_from_two_epocs = np.zeros(len(num_labels), np.int32)
        expected_labels=np.sum(train_labels,0)
        
        for step in range(26):
            feats, labels = miniBatchReader.get_next_minibatch()
            labels_from_two_epocs += np.sum(labels,0)
        for idx, val in enumerate(labels_from_two_epocs):
            assert val == 2*expected_labels[idx], "val=%d expected=%d label=%d" % (val, 2*expected_labels[idx], idx)

    def tearDown(self) :
        for h5filename in self.h5filenames:
            os.unlink(h5filename)

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])

