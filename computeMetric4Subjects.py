import os
import numpy as np
from utils import dice,IoU
import SimpleITK as sitk

def main():
    path_test = '/home/dongnie/Desktop/myPyTorch/pytorch-SRResNet/res_1106'
    ids = [1,2,3,4,6,7,8,10,11,12,13,29]
#     ids = [1,2,29]
    for ind in ids:    
#         mr_test_itk = sitk.ReadImage(os.path.join(path_test,'denseCrf3dSegmMap_pelvic_sub%d.nii.gz'%ind))
        mr_test_itk = sitk.ReadImage(os.path.join(path_test,'preSub_wce_wdice_adImpo_viewExp_1106_sub%d.nii.gz'%ind))
        crfSeg = sitk.GetArrayFromImage(mr_test_itk)
        ct_test_itk = sitk.ReadImage(os.path.join(path_test,'gt_sub%d.nii.gz'%ind))
        gtSeg = sitk.GetArrayFromImage(ct_test_itk)
                
        diceBladder = dice(crfSeg,gtSeg,1)
        diceProstate = dice(crfSeg,gtSeg,2)
        diceRectumm = dice(crfSeg,gtSeg,3)
     
        iouBladder = IoU(crfSeg,gtSeg,1)
        iouProstate = IoU(crfSeg,gtSeg,2)
        iouRectum = IoU(crfSeg,gtSeg,3)
        print 'sub%d'%ind,' iou1 = ',iouBladder,' iou2= ',iouProstate,' iou3= ',iouRectum
     
        print 'sub%d'%ind, 'dice1 = ',diceBladder,' dice2= ',diceProstate,' dice3= ',diceRectumm 
        print '\n'   
if __name__ == '__main__':
#     testGradients()     
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()