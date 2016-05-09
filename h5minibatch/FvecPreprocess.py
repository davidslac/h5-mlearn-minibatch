import os
import numpy as np
import h5py

def loadFvecStats(fname):
    '''Loads the fvec stats from file written by Samples

    ARGS:
      fname - name of h5 file to write/read stats from
    '''
    h5 = h5py.File(fname,'r')
    fvecstats = {}
    for ky in h5.keys(): 
        fvecstats[ky] = h5[ky].value
    h5.close()
    return fvecstats


def preprocessFvecBatch(fvec, fvecstats, datasets_for_feature_vector, eps=1e-6):
    '''whiten a batch of feature vectors
    ARGS:
      fvec - batch of features, a 2D numpy array of a numeric type, each column 
             corresponds to a name in the datasets_for_feature_vector list
      datasets_for_feature_vector - list of feature names, one for each column
                                    of fvec
      fvecstats - dictionary for all bld stats, could be more than whats in 
                  fvec. For each name in datasets_for_feature_vector, 
                  includes nm:mean and nm:std, these are used to whiten
                  fvec
      eps - added to stddev normalization for numerical stability

    RET:
     whittened fvec
    '''
    assert fvec.shape[1]==len(datasets_for_feature_vector)
    fvecWhittened = np.zeros(fvec.shape, fvec.dtype)
    for idx, nm in enumerate(datasets_for_feature_vector):
        mean = fvecstats['%s:mean'%nm]
        stddev = fvecstats['%s:std' % nm]
        fvecWhittened[:,idx] = (fvec[:,idx]-mean)/(eps+stddev)
    return fvecWhittened
