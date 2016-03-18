from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import copy
import time
import warnings

import numpy as np

from .SamplesCache import SamplesCache
from .utils import get_all_samples
from .utils import get_balanced_samples
from .utils import load_features_and_labels


class H5MiniBatchReader(object):
    '''delivers minibatches of feature/labels from a collection of h5 files
    '''
    def __init__(self, 
                 h5files,
                 minibatch_size,
                 validation_size,
                 feature_dataset='features',
                 label_dataset='labels',
                 return_as_one_hot=False,
                 number_of_batches=None,
                 feature_preprocess=None,
                 class_labels_max_imbalance_ratio=None,
                 max_mb_to_preload_all=None,  # can be numeric
                 random_seed=None,
                 add_channel_to_2D=None, # can be "row_col_channel", or "channel_row_col"
                 verbose=False):

        if random_seed is not None:
            np.random.seed(random_seed)
        _h5files = copy.deepcopy(h5files)
        original_len_h5files = len(_h5files)
        _h5files = list(set(_h5files))
        if len(_h5files) != original_len_h5files:
            warnings.warn("H5MiniBatchReader: duplicate filenames "
                          "given in h5files argument - uniqifying")
        _h5files.sort()
        self._h5files = _h5files
        self._minibatch_size = minibatch_size
        self._validation_size = validation_size
        self._feature_dataset = feature_dataset
        self._label_dataset = label_dataset
        self._return_as_one_hot = return_as_one_hot
        self._number_of_batches = number_of_batches
        self._feature_preprocess = feature_preprocess
        self._class_labels_max_imbalance_ratio = class_labels_max_imbalance_ratio
        self._max_mb_to_preload_all = max_mb_to_preload_all
        self._add_channel_to_2D = add_channel_to_2D
        self._verbose = verbose

        if self._verbose:
            t0 = time.time()
            sys.stdout.write("Scanning through h5 files ...\n")
            sys.stdout.flush()
        all_samples = get_all_samples(self._h5files, self._label_dataset, self._verbose)
        # randomize all_samples before potentially grabbing a small number of samples for training
        np.random.shuffle(all_samples)
        if self._verbose:
            sys.stdout.write("... scan took %.2f sec\n" % (time.time()-t0,))
            sys.stdout.flush()
        self._num_outputs = np.max(all_samples[:,2])+1

        if self._class_labels_max_imbalance_ratio is not None:
            balanced_samples = get_balanced_samples(all_samples, 
                                                    self._num_outputs, 
                                                    self._class_labels_max_imbalance_ratio)
            if self._verbose:
                sys.stdout.write(("After balancing samples per ratio=%.2f, " + \
                                 "there are %d samples available for train/validation\n") % \
                                 (self._class_labels_max_imbalance_ratio,
                                  len(balanced_samples)))
                sys.stdout.flush()
            # balancing produces samples grouped by label. randomize before we 
            # grab validation set
            np.random.shuffle(balanced_samples)

            all_samples = balanced_samples

        total_samples = len(all_samples)
        assert total_samples > self._validation_size, ("validation_size=%d " + \
            "is greater than number of samples available=%d") % (self._validation_size,
                                                                 total_samples)

        validation_samples = all_samples[0:self._validation_size,:]

        train_samples = all_samples[self._validation_size:,:]

        assert len(train_samples) > self._minibatch_size, \
        "one minibatch is greater than the available train samples"
        
        if self._number_of_batches:
            assert self._number_of_batches * self._minibatch_size <= len(train_samples), \
                ("after validation and potential balancing, available train samples is %d, but " + \
                 "minibatch_size(=%d) * number_of_batches(=%d) is %d" ) % \
                (len(train_samples), self._minibatch_size, self._number_of_batches,
                 self._minibatch_size * self._number_of_batches)
            
        if not self._number_of_batches:
            self._number_of_batches = len(train_samples)//self._minibatch_size

        number_of_train_samples = self._number_of_batches * self._minibatch_size
        if number_of_train_samples < len(train_samples) and self._verbose:
            sys.stdout.write("not using %d samples\n" % (len(train_samples)-number_of_train_samples))
            sys.stdout.flush()
        self._train_samples = train_samples[0:number_of_train_samples,:]
        assert len(self._train_samples)>0
        self._set_features_shape()

        self._train_next_minibatch_idx = 0

        t0 = time.time()
        self._one_hot_arg = None
        if self._return_as_one_hot:
            self._one_hot_arg = self._num_outputs

        if self._validation_size>0:
            self._validation_features, self._validation_labels = \
                load_features_and_labels(dataset=self._feature_dataset,
                                         samples=validation_samples, 
                                         h5files=self._h5files, 
                                         preprocess=self._feature_preprocess,
                                         add_channel_to_2D=self._add_channel_to_2D,
                                         one_hot_num_outputs=self._one_hot_arg)
        else:
            self._validation_features = None
            self._validation_labels = None

        if self._verbose and (self._validation_labels is not None):
            sys.stdout.write("Read %d test samples in %.2f sec\n" % 
                             (len(self._validation_labels),time.time()-t0))
            sys.stdout.flush()

        self._train_samples_cache = self._try_to_preload_training(self._max_mb_to_preload_all,
                                                                 self._verbose)

    def _set_features_shape(self):
        feats, labels =load_features_and_labels(dataset=self._feature_dataset,
                                                samples=self._train_samples[0:1,:],
                                                h5files=self._h5files,
                                                preprocess=None,
                                                add_channel_to_2D=self._add_channel_to_2D,
                                                one_hot_num_outputs=None)
        self._features_shape = feats.shape[1:]
        self._features_dtype = feats.dtype


    def _try_to_preload_training(self, max_mb_to_preload_all, verbose):
        if not max_mb_to_preload_all:
            return None
            
        bytesPerPreprocessedPixel = 4.0
        mbToStoreTrain = bytesPerPreprocessedPixel
        for rnk in self._features_shape:
            mbToStoreTrain *= rnk
        mbToStoreTrain *= len(self._train_samples)
        mbToStoreTrain /= float(1<<20)
        if mbToStoreTrain < max_mb_to_preload_all:
            t0 = time.time()
            sys.stdout.write("starting to preload %.2fMB of data ...\n" % mbToStoreTrain)
            sys.stdout.flush()
            features, labels = load_features_and_labels(dataset=self._feature_dataset,
                                                        samples=self._train_samples,
                                                        h5files=self._h5files, 
                                                        preprocess=self._feature_preprocess,
                                                        add_channel_to_2D=self._add_channel_to_2D,
                                                        one_hot_num_outputs=self._one_hot_arg)
            sys.stdout.write("preloading data took %.2f sec.\n" % (time.time()-t0,))
            sys.stdout.flush()
            return SamplesCache(self._train_samples, features, labels)
        if max_mb_to_preload_all > 0 and verbose:
            print("Did not preload training. Needed %.2f MB, but limit was %.2f" % 
                  (mbToStoreTrain, max_mb_to_preload_all))
        return None

    def features_placeholder_shape(self):
        placeholder_shape = [None,]
        for rnk in self._features_shape:
            placeholder_shape.append(rnk)
        return placeholder_shape
    
    def num_outputs(self):
        return self._num_outputs

    def get_validation_set(self):
        return self._validation_features, self._validation_labels

    def get_next_minibatch(self):
        if self._train_next_minibatch_idx >= len(self._train_samples):
            self._train_next_minibatch_idx = 0
            np.random.shuffle(self._train_samples)
        first=self._train_next_minibatch_idx
        last=first + self._minibatch_size
        samples_to_load = self._train_samples[first:last,:]
        self._train_next_minibatch_idx += self._minibatch_size
        features, labels = self._load_features_check_cache(samples_to_load)
        return features, labels

    def _load_features_check_cache(self, samples_to_load):
        if self._train_samples_cache:
            return self._train_samples_cache.get_samples(samples_to_load)
            
        features, labels = load_features_and_labels(dataset=self._feature_dataset,
                                                    samples=samples_to_load, 
                                                    h5files=self._h5files, 
                                                    preprocess=self._feature_preprocess,
                                                    add_channel_to_2D=self._add_channel_to_2D,
                                                    one_hot_num_outputs=self._one_hot_arg)
        return features, labels

    def get_all_train(self):
        if self._train_samples_cache:
            return self._train_samples_cache.get_samples(None)
            
        features, labels = load_features_and_labels(dataset=self._feature_dataset,
                                                    samples=self._train_samples,
                                                    h5files=self._h5files, 
                                                    preprocess=self._feature_preprocess,
                                                    add_channel_to_2D=self._add_channel_to_2D,
                                                    one_hot_num_outputs=self._one_hot_arg)
        return features, labels
