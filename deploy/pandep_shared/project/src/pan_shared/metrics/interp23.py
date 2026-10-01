"""Evaluation/data-audit only. DLPan interp23tap, GPL-3.0; see vendor_reference/LICENSES.
Do not include this module in inference bundles.
"""
import numpy as np
import math
import scipy.ndimage.filters as ft

def interp23tap(img, ratio):

    assert((2**(round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'

    r,c,b = img.shape

    CDF23 = np.asarray([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0, -0.000060081482])
    CDF23 = [element * 2 for element in CDF23]
    BaseCoeff = np.expand_dims(np.concatenate([np.flip(CDF23[1:]), CDF23]), axis=-1)


    for z in range(int(ratio/2)):

        I1LRU = np.zeros(((2 ** (z+1)) * r, (2 ** (z+1)) * c, b))

        if z == 0:
            I1LRU[1::2, 1::2,:] = img
        else:
            I1LRU [::2,::2,:] = img

        for i in range(b):
            temp = ft.convolve(np.transpose(I1LRU[:,:,i]), BaseCoeff, mode='wrap')
            I1LRU[:, :, i] = ft.convolve(np.transpose(temp), BaseCoeff, mode='wrap')

        img = I1LRU

    return img
