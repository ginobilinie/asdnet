import os
import h5py
import SimpleITK as sitk
import cv2
import numpy as np
import scipy.ndimage as nd
import scipy.io as scio

'''
obtain 0/1 organ maps for the specified organID
Input:
    img: the original image map
    organID: the specified organID
Output:
    the 0/1 feature map for the specified organID
'''
def obtainSingleOrganMap(img, organID):
    i,j,k = np.where(img==organID)
    img1 = np.zeros(img.shape,dtype=int)
    img1[i,j,k] = 1
    return img1

'''
using morphological transformation (e.g., erosion, dilation) to deal with the images
'''    
def denoiseImg_closing(img, kernel):
    img1 = obtainSingleOrganMap(img, 1)
    img2 = obtainSingleOrganMap(img, 2)
    img3 = obtainSingleOrganMap(img, 3)
#     print img1.shape
#     closing1 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)    
#     closing2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)     
#     closing3 = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, kernel)   
     
    out1 = nd.binary_closing(img1, structure=np.ones((10,10,10)))
    out2 = nd.binary_closing(img2, structure=np.ones((3,3,3)))
    out3 = nd.binary_closing(img3, structure=np.ones((10,10,10)))  
    
#     out1 = nd.binary_opening(img1, structure=np.ones((5,5,5)))
#     out2 = nd.binary_opening(img2, structure=np.ones((3,3,3)))
#     out3 = nd.binary_opening(img3, structure=np.ones((10,10,10)))
#     
#     out1 = nd.binary_erosion(out1, structure=np.ones((2,2,2)))
    out2 = nd.binary_erosion(out2, structure=np.ones((2,2,2)))
#     out3 = nd.binary_closing(out3, structure=np.ones((5,5,5)))

    output = np.zeros(img.shape)
    
    i,j,k = np.where(out1==1)
    output[i,j,k] = 1
    i,j,k = np.where(out2==1)
    output[i,j,k] = 2
    i,j,k = np.where(out3==1)
    output[i,j,k] =3
    
    return output

'''
 Return array with completely isolated single cells removed
:param array: Array with completely isolated single cells
:param struct: Structure array for generating unique regions
:return: Array with minimum region size > 1
'''
def filter_isolated_cells(array, struct):

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
denoise Images for each unique intensity, we remove the isolated regions with given struct (kernels)
'''
def denoiseImg_isolation(array, struct):
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

'''
this function is used to compute the dice ratio
'''
def dice(im1, im2,tid):
    im1=im1==tid #make it boolean
    im2=im2==tid #make it boolean
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc

def main():
    path='/home/dongnie/Desktop/myPyTorch/pytorch-SRResNet/'
    saveto='/home/dongnie/Desktop/myPyTorch/pytorch-SRResNet/'
    caffeApp=0
    fileIDs=[1]
    #fileIDs=[1]
    for ind in fileIDs:
        datafilename='preSub%d_clean_wdice_wce_1016_150000.nii.gz'%ind #provide a sample name of your filename of data here
        datafn=os.path.join(path,datafilename)
        labelfilename='gt/gt%d.nii'%ind  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(path,labelfilename)
        prefilename='preSub%d_mt1_wdice_wce_1016_150000.nii.gz'%ind #provide a sample name of your filename of data here
        prefn=os.path.join(path,prefilename)

        imgOrg=sitk.ReadImage(datafn)
        mrimg=sitk.GetArrayFromImage(imgOrg)
        labelOrg=sitk.ReadImage(labelfn)
        labelimg=sitk.GetArrayFromImage(labelOrg) 
        # Run function on sample array
        #filtered_array = denoiseImg_isolation(mrimg, struct=np.ones((3,3,3)))        
        filtered_array = denoiseImg_closing(mrimg, kernel=np.ones((20,20,20)))        
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