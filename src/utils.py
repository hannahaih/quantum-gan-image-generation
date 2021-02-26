# Utility functions

# https://stackoverflow.com/questions/15505514/binary-numpy-array-to-list-of-integersd

def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y


# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697

import contextlib
import joblib
from tqdm import tqdm    
from joblib import Parallel, delayed

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding

import numpy as np

def vec_bin_array(arr, m):
    """
    Arguments: 
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[...,bit_ix] = fetch_bit_func(strs).astype("int8")

    return ret


import plotly.express as px

def plot_dist(dist, figure_width=500):
    return px.imshow(dist, width=figure_width, height=figure_width, color_continuous_scale='sunset')


from scipy.ndimage.filters import gaussian_filter

def blur_and_normalize(array, sigma=0.0, show_figure=False, figure_width=500):
    """
    Preprocess input image with gaussian filter
    Args:
        array: input array from the image
        sigma (scalar): std for the gaussian kernel
        show_figure (boolean): True for plotting the processed image
        figure_width (scalar)
    Returns:
        dist (array): output array from preprocessed image
    """
    blurred = gaussian_filter(array, sigma=sigma)
    dist = blurred / np.sum(blurred)
    
    if show_figure:
        fig = plot_dist(dist, figure_width=figure_width)
        fig.show()
    
    return dist