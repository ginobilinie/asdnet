'''
if a image (e.g., segmentation maps) is composed of several components, but there are some noise, which are in form of isolated components
we can then remove the extra noise with following codes
Dong Nie
12/17/2016
'''
import numpy as np
import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import scipy.io as scio
from scipy import ndimage as nd
from imgUtils import dice


def filter_isolated_cells(array, struct):
    """ Return array with completely isolated single cells removed
    :param array: Array with completely isolated single cells
    :param struct: Structure array for generating unique regions
    :return: Array with minimum region size > 1
    """

    filtered_array = np.copy(array)
    #id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_regions, num_ids = nd.measurements.label(filtered_array, structure=struct)
    #print 'id_region shape is ',id_regions.shape
    #print 'num_ids is ',num_ids
    #id_regions:unique label for unique features
    #num_features: how many objects are found
    #id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    id_sizes = np.array(nd.measurements.sum(array, id_regions, range(num_ids + 1))) #number of pixels for this region (id)
    #An array of the sums of values of input inside the regions defined by labels with the same shape as index. If 'index' is None or scalar, a scalar is returned.
    #print 'id_sizes shape is ',id_sizes.shape
    #print 'id_sizes is ', id_sizes
    maxV=np.amax(id_sizes) 
    for v in id_sizes:
        if v==maxV:
            continue
        area_mask = (id_sizes == v)
        #print 'area_mask.shape is ', area_mask.shape
        filtered_array[area_mask[id_regions]] = 0
    return filtered_array

'''
denoise Images for each unique intensity 
'''
def denoiseImg(array, struct):
    uniqueVs=np.unique(array)
    denoised_array=np.zeros(array.shape)
    for v in uniqueVs:
        temp_array=np.zeros(array.shape)
        vMask=(array==v)
        temp_array[vMask]=v
        #print 'vMask shape, ',vMask.shape
        #print 'arrayV shape, ',arrayV.shape
        filtered_array=filter_isolated_cells(temp_array,struct)
        denoised_array[(filtered_array==v)]=v
    return denoised_array 


def main():
    path='/home/dongnie/Desktop/Caffes/caffe/python/pelvicSeg/'
    saveto='/home/dongnie/Desktop/Caffes/caffe/python/pelvicSeg/'
    caffeApp=0
    fileIDs=[1,2,3,4,6,7,8,10,11,12,13]
    #fileIDs=[1]
    for ind in fileIDs:
        datafilename='preSub%d_5x168x112.nii'%ind #provide a sample name of your filename of data here
        datafn=os.path.join(path,datafilename)
        labelfilename='gt%d.nii'%ind  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(path,labelfilename)
        prefilename='preSub%d_denoised.nii'%ind #provide a sample name of your filename of data here
        prefn=os.path.join(path,prefilename)

        imgOrg=sitk.ReadImage(datafn)
        mrimg=sitk.GetArrayFromImage(imgOrg)
        labelOrg=sitk.ReadImage(labelfn)
        labelimg=sitk.GetArrayFromImage(labelOrg) 
        # Run function on sample array
        #filtered_array = filter_isolated_cells(mrimg, struct=np.ones((3,3,3)))        
        filtered_array = denoiseImg(mrimg, struct=np.ones((3,3,3)))        
        # Plot output, with all isolated single cells removed
        #plt.imshow(filtered_array, cmap=plt.cm.gray, interpolation='nearest')
        pr0=dice(labelimg,filtered_array,0)
        pr1=dice(labelimg,filtered_array,1)
        pr2=dice(labelimg,filtered_array,2)
        pr3=dice(labelimg,filtered_array,3)
        print 'dice for sub%d: '%ind,pr0, ' ',pr1,' ',pr2,' ',pr3
        preVol=sitk.GetImageFromArray(filtered_array)
        sitk.WriteImage(preVol,prefn)
if __name__ == '__main__':     
    main()
