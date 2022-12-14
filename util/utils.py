import random
import numpy as np
import warnings
from scipy.signal import correlate2d


def introduce_defect(img, total_defective_pixels, padding):
    
    """
    This function randomly replaces pixels values with extremely high or low pixel values to create dead pixels (Dps).
    Note that the defective pixel values are never introduced on the periphery of the image to ensure that there 
    are no adjacent DPs. 
    
    Parameters
    ----------
    img: 2D ndarray
    total_defective_pixels: number of Dps to introduce in img.
    padding: bool value (set to True to add padding)
    
    Returns
    -------
    defective image: image/padded img containing specified (by TOTAL_DEFECTIVE_PIXELS) 
    number of dead pixels.
    orig_val: ndarray of size same as img/padded img containing original pixel values in place of introduced DPs 
    and zero elsewhere. 
    """
    
    if padding:
        padded_img = np.pad(img, ((2,2), (2,2)), "reflect")
    else:
        padded_img  = img.copy()
   
    orig_val   = np.zeros((padded_img.shape[0], padded_img.shape[1])) 
    
    while total_defective_pixels:
        defect     = [random.randrange(1,15), random.randrange(4081, 4095)]   # stuck low int b/w 1 and 15, stuck high float b/w 4081 and 4095
        defect_val = defect[random.randint(0,1)] 
        random_row, random_col   = random.randint(2, img.shape[0]-3), random.randint(2, img.shape[1]-3)
        left, right  = orig_val[random_row, random_col-2], orig_val[random_row, random_col+2]
        top, bottom  = orig_val[random_row-2, random_col], orig_val[random_row+2, random_col]
        neighbours   = [left, right, top, bottom]
        
        if not any(neighbours) and orig_val[random_row, random_col]==0: # if all neighbouring values in orig_val are 0 and the pixel itself is not defective
            orig_val[random_row, random_col]   = padded_img[random_row, random_col]
            padded_img[random_row, random_col] = defect_val
            total_defective_pixels-=1
    
    return padded_img, orig_val

def gaussKernRAW(N, stdDev, stride):
    """
    This function takes in size, standard deviation and spatial stride required for adjacet weights to output a gaussian kernel of size NxN

    Parameters
    ----------
    N:      size of gaussian kernel, odd
    stdDev: standard deviation of the gaussian kernel
    stride: spatial stride between to be considered for adjacent gaussian weights
    
    Returns
    -------
    outKern: an output gaussian kernel of size NxN
    """

    if N%2 == 0:
        warnings.warn('kernel size (N) cannot be even, setting it as odd value')
        N = N + 1

    if N <= 0:
        warnings.warn('kernel size (N) cannot be <= zero, setting it as 3')
        N = 3
        
    outKern = np.zeros((N,N), dtype=np.float32)

    for i in range(0,N):
        for j in range (0,N):
            outKern[i,j] = np.exp(-1 * ((stride*(i - ((N-1)/2)))**2 + (stride*(j - ((N-1)/2)))**2) / (2 * (stdDev**2)))
        
    sumKern = np.sum(outKern)
    outKern[0:N:1, 0:N:1] = outKern[0:N:1, 0:N:1] / sumKern

    return outKern    

def crop(img, rows_to_crop=0, cols_to_crop=0):
        
    """
    Crop 2D array.
    Parameter:
    ---------
    img: image (2D array) to be cropped.
    rows_to_crop: Number of rows to crop. If it is an even integer, 
                    equal number of rows are cropped from either side of the image. 
                    Otherwise the image is cropped from the extreme right/bottom.
    cols_to_crop: Number of columns to crop. Works exactly as rows_to_crop.
    
    Output: cropped image
    """
    
    if rows_to_crop:
        if rows_to_crop%2==0:
            img = img[rows_to_crop//2:-rows_to_crop//2, :]
        else:
            img = img[0:-1, :]
    if cols_to_crop:         
        if cols_to_crop%2==0:
            img = img[:, cols_to_crop//2:-cols_to_crop//2]
        else:
            img = img[:, 0:-1] 
    return img

def stride_convolve2d(matrix, kernel):
    return correlate2d(matrix, kernel, mode="valid")[::kernel.shape[0], ::kernel.shape[1]]        