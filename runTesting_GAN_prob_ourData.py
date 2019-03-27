# from __future__ import print_function
import argparse, os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch
import torch.utils.data as data_utils
from utils import *
from ganComponents import *
from nnBuildUnits import CrossEntropy2d
from nnBuildUnits import computeSampleAttentionWeight
from nnBuildUnits import adjust_learning_rate
import time
from morpologicalTransformation import denoiseImg_closing,denoiseImg_isolation

# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--NDim", type=int, default=3, help="the dimension of the shape, 1D, 2D or 3D?")
parser.add_argument("--in_channels", type=int, default=1, help="the input channels ?")
parser.add_argument("--out_channels", type=int, default=4, help="the output channels (num of classes)?")
parser.add_argument("--in_channels_netD", type=int, default=4, help="the input channels for netD?")
parser.add_argument("--out_channels_netD", type=int, default=2, help="the output channels for netD?")
parser.add_argument("--input_sz", type=arg_as_list, default=[16,64,64], help="the input patch size of list")
parser.add_argument("--output_sz", type=arg_as_list, default=[16,64,64], help="the output patch size of list")
parser.add_argument("--test_step_sz", type=arg_as_list, default=[1,4,4], help="the step size at testing one subject")
parser.add_argument("--isSegReg", action="store_true", help="is Seg and Reg?", default=False)
parser.add_argument("--isDiceLoss", action="store_true", help="is Dice Loss used?", default=True)
parser.add_argument("--isSoftmaxLoss", action="store_true", help="is Softmax Loss used?", default=True)
parser.add_argument("--isContourLoss", action="store_true", help="is Contour Loss used?", default=False)
parser.add_argument("--isResidualEnhancement", action="store_true", help="is residual learning operation enhanced?", default=False)
parser.add_argument("--isViewExpansion", action="store_true", help="is view expanded?", default=True)
parser.add_argument("--isAdLoss", action="store_true", help="is adversarial loss used?", default=True)
parser.add_argument("--isNetDFullyConv", action="store_true", help="is the netD fully convolutional network?", default=True)
parser.add_argument("--isNetDInputIncludeSource", action="store_true", help="is the input of netD including the source image or not?", default=False)
parser.add_argument("--isSpatialDropOut", action="store_true", help="is spatial dropout used?", default=False)
parser.add_argument("--isFocalLoss", action="store_true", help="is focal loss used?", default=False)
parser.add_argument("--isSampleImportanceFromAd", action="store_true", help="is sample importance from adversarial network used?", default=False)
parser.add_argument("--dropoutRate", type=float, default=0.25, help="Spatial Dropout Rate. Default=0.25")
parser.add_argument("--lambdaAD", type=float, default=0, help="loss coefficient for AD loss. Default=0")
parser.add_argument("--adImportance", type=float, default=0, help="Sample importance from AD network. Default=0")

parser.add_argument("--batchSize", type=int, default=10, help="training batch size")
parser.add_argument("--numofIters", type=int, default=200000, help="number of iterations to train for")
parser.add_argument("--showTrainLossEvery", type=int, default=100, help="number of iterations to show train loss")
parser.add_argument("--saveModelEvery", type=int, default=5000, help="number of iterations to save the model")
parser.add_argument("--showTestPerformanceEvery", type=int, default=1000, help="number of iterations to show test performance")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--decLREvery", type=int, default=100000, help="Sets the learning rate to the initial LR decayed by momentum every n iterations, Default: n=40000")
parser.add_argument("--cuda", action="store_true", help="Use cuda?", default=True)
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--prefixModelName", default="/shenlab/lab_stor5/dongnie/pelvic/Seg3D_wce_viewExp_resEnhance_iter50000_0224_ourData", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--test_input_file_name",default='img1_nocrop.nii.gz',type=str, help="the input file name for testing subject")
parser.add_argument("--test_gt_file_name",default='img1_label_nie_nocrop.nii.gz',type=str, help="the ground-truth file name for testing subject")

parser.add_argument("--modelPath", default="/shenlab/lab_stor5/dongnie/Seg3D_wdice_wce_viewExp_resEnhance_fullAD0p05_0224_ourData_55000.pt", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--netDModelPath", default="/shenlab/lab_stor5/dongnie/Seg3D_wdice_wce_viewExp_resEnhance_fullAD0p05_0224_ourData_netD_55000.pt", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="Seg3D_wdice_wce_viewExp_resEnhance_fullAD0p05_netD_iter55000_0224_ourData", type=str, help="prefix of the to-be-saved predicted filename")
parser.add_argument("--how2normalize", type=int, default=4, help="how to normalize the data")
parser.add_argument("--resType", type=int, default=2, help="resType: 0: segmentation map (integer); 1: regression map (continuous); 2: segmentation map + probability map")

'''
This script tests the segmenter as well as the discriminator
'''

def main():
    global opt
    opt = parser.parse_args()
    print opt
    
    path_test = '/home/dongnie/warehouse/pelvicSeg/prostateChallenge/data/'
    path_test = '/shenlab/lab_stor5/dongnie/challengeData/data/'
    path_test = '/shenlab/lab_stor5/dongnie/pelvic/'
    if opt.isSegReg:
        negG = ResSegRegNet(opt.in_channels, opt.out_channels, nd=opt.NDim)
    elif opt.isContourLoss:
        netG = ResSegContourNet(opt.in_channels, opt.out_channels, nd=opt.NDim, isRandomConnection=opt.isResidualEnhancement,isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut,dropoutRate=opt.dropoutRate)
    else:
        netG = ResSegNet(opt.in_channels, opt.out_channels, nd=opt.NDim, isRandomConnection=opt.isResidualEnhancement,isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut,dropoutRate=opt.dropoutRate)
    #netG.apply(weights_init)
    
    if opt.isAdLoss:
        if opt.isNetDFullyConv:
            netD = Discriminator_my23DLRResFCN(opt.in_channels_netD, opt.out_channels_netD, nd=opt.NDim)
        else:
            netD = Discriminator(opt.in_channels_netD, opt.out_channels_netD, nd=opt.NDim)
        netD = netD.cuda()
        netD.load_state_dict(torch.load(opt.netDModelPath))

            
            
    netG = netG.cuda()
    
    checkpoint = torch.load(opt.modelPath)
#     netG.load_state_dict(checkpoint["model"].state_dict())
    netG.load_state_dict(checkpoint["model"])
#     netG.load_state_dict(torch.load(opt.modelPath))
    
    ids = [1,2,3,4,6,7,8,10,11,12,13]
#     ids = [45,46,47,48,49]
    ids = [1,2,3,4,13,29]
    for ind in ids:
#         mr_test_itk=sitk.ReadImage(os.path.join(path_test,'Case%d.nii.gz'%ind))
#         ct_test_itk=sitk.ReadImage(os.path.join(path_test,'Case%d_segmentation.nii.gz'%ind))
        mr_test_itk=sitk.ReadImage(os.path.join(path_test,'img%d_nocrop.nii.gz'%ind))
        ct_test_itk=sitk.ReadImage(os.path.join(path_test,'img%d_label_nie_nocrop.nii.gz'%ind))
        
        mrnp=sitk.GetArrayFromImage(mr_test_itk)
        mu=np.mean(mrnp)
    
        ctnp=sitk.GetArrayFromImage(ct_test_itk)
        
        #for training data in pelvicSeg
        if opt.how2normalize == 1:
            maxV, minV=np.percentile(mrnp, [99 ,1])
            print 'maxV,',maxV,' minV, ',minV
            mrnp=(mrnp-mu)/(maxV-minV)
            print 'unique value: ',np.unique(ctnp)
    
        #for training data in pelvicSeg
        elif opt.how2normalize == 2:
            maxV, minV = np.percentile(mrnp, [99 ,1])
            print 'maxV,',maxV,' minV, ',minV
            mrnp = (mrnp-mu)/(maxV-minV)
            print 'unique value: ',np.unique(ctnp)
        
        #for training data in pelvicSegRegH5
        elif opt.how2normalize== 3:
            std = np.std(mrnp)
            mrnp = (mrnp - mu)/std
            print 'maxV,',np.ndarray.max(mrnp),' minV, ',np.ndarray.min(mrnp)
            
        elif opt.how2normalize== 4:
                maxV, minV = np.percentile(mrnp, [99.2 ,1])
                print 'maxV is: ',np.ndarray.max(mrnp)
                mrnp[np.where(mrnp>maxV)] = maxV
                print 'maxV is: ',np.ndarray.max(mrnp)
                mu=np.mean(mrnp)
                std = np.std(mrnp)
                mrnp = (mrnp - mu)/std
                print 'maxV,',np.ndarray.max(mrnp),' minV, ',np.ndarray.min(mrnp)
    
    #             full image version with average over the overlapping regions
    #             ct_estimated = testOneSubject(mrnp,ctnp,[3,168,112],[1,168,112],[1,8,8],netG,'Segmentor_model_%d.pt'%iter)
        
        # the attention regions
#         x1=80
#         x2=192
#         y1=35
#         y2=235
#         matFA = mrnp[:,y1:y2,x1:x2] #note, matFA and matFAOut same size 
#         matGT = ctnp[:,y1:y2,x1:x2]
        matFA = mrnp
        matGT = ctnp
        
        if opt.resType==2:
            matOut, matProb, _ = testOneSubject(matFA,matGT,opt.out_channels,opt.input_sz,opt.output_sz,opt.test_step_sz,netG,opt.modelPath,resType=opt.resType, nd = opt.NDim)
        else:
            matOut,_ = testOneSubject(matFA,matGT,opt.out_channels,opt.input_sz,opt.output_sz,opt.test_step_sz,netG,opt.modelPath,resType=opt.resType, nd = opt.NDim)
                                      
        #matOut,_ = testOneSubject(matFA,matGT,opt.out_channels,opt.input_sz, opt.output_sz, opt.test_step_sz,netG,opt.modelPath, nd = opt.NDim)
        ct_estimated = np.zeros([ctnp.shape[0],ctnp.shape[1],ctnp.shape[2]])
        ct_prob = np.zeros([opt.out_channels, ctnp.shape[0],ctnp.shape[1],ctnp.shape[2]])
#         print 'matOut shape: ',matOut.shape
#         ct_estimated[:,y1:y2,x1:x2] = matOut
        ct_estimated = matOut
        ct_prob = matProb
        matProb_Bladder = np.squeeze(ct_prob[1,:,:,:])
        matProb_Prostate = np.squeeze(ct_prob[2,:,:,:])
        matProb_Rectum = np.squeeze(ct_prob[3,:,:,:])
        
        threshold = 0.3
        tmat_prob = np.zeros(matProb_Bladder.shape)
        tmat = np.zeros(matProb_Bladder.shape)
        
        #for bladder
        inds1 = np.where(matProb_Bladder>threshold)
        tmat_prob[inds1] = matProb_Bladder[inds1]
        tmat[inds1] = 1
        #for prostate
        inds2 = np.where(matProb_Prostate>threshold)
        tmat_prob[inds2] = matProb_Prostate[inds2]
        tmat[inds2] = 2  
        #for rectum
        inds3 = np.where(matProb_Rectum>threshold)
        tmat_prob[inds3] = matProb_Rectum[inds3]
        tmat[inds3] = 3           
        
        tmat = denoiseImg_closing(tmat, kernel=np.ones((20,20,20))) 
        tmat = denoiseImg_isolation(tmat, struct=np.ones((3,3,3)))       
        diceBladder = dice(tmat,ctnp,1)
        diceProstate = dice(tmat,ctnp,2)
        diceRectum = dice(tmat,ctnp,3)
        print 'sub%d'%ind,'dice1 = ',diceBladder,' dice2= ',diceProstate,' dice3= ',diceRectum
                               
        volout = sitk.GetImageFromArray(matProb_Prostate)
        sitk.WriteImage(volout,opt.prefixPredictedFN+'_probmap_Prostate_sub{:02d}'.format(ind)+'.nii.gz')
        volout = sitk.GetImageFromArray(matProb_Bladder)
        sitk.WriteImage(volout,opt.prefixPredictedFN+'_probmap_Bladder_sub{:02d}'.format(ind)+'.nii.gz')  
        volout = sitk.GetImageFromArray(matProb_Rectum)
        sitk.WriteImage(volout,opt.prefixPredictedFN+'_probmap_Rectum_sub{:02d}'.format(ind)+'.nii.gz')    

        volout = sitk.GetImageFromArray(tmat_prob)
        sitk.WriteImage(volout,opt.prefixPredictedFN+'threshold0p3_probmap_sub{:02d}'.format(ind)+'.nii.gz')  
        
        volout = sitk.GetImageFromArray(tmat0p3)
        sitk.WriteImage(volout,opt.prefixPredictedFN+'threshold0p3_segmap_sub{:02d}'.format(ind)+'.nii.gz')
        
                   
        ct_estimated = np.rint(ct_estimated) 
        ct_estimated = denoiseImg_closing(ct_estimated, kernel=np.ones((20,20,20))) 
        ct_estimated = denoiseImg_isolation(ct_estimated, struct=np.ones((3,3,3)))   
        
        
    
#         diceProstate = dice(ct_estimated,ctnp,2)
#         diceRectumm = dice(ct_estimated,ctnp,3)
        
#         print 'pred: ',ct_estimated.dtype, ' shape: ',ct_estimated.shape
#         print 'gt: ',ctnp.dtype,' shape: ',ctnp.shape
#         print 'sub%d'%ind,'dice1 = ',diceBladder,' dice2= ',diceProstate,' dice3= ',diceRectumm
        diceBladder = dice(ct_estimated,ctnp,1)
        diceProstate = dice(ct_estimated,ctnp,2)
        diceRectum = dice(ct_estimated,ctnp,3)
        print 'sub%d'%ind,'dice1 = ',diceBladder,' dice2= ',diceProstate,' dice3= ',diceRectum
        volout = sitk.GetImageFromArray(ct_estimated)
        sitk.WriteImage(volout,opt.prefixPredictedFN+'sub{:02d}'.format(ind)+'.nii.gz')
        volgt = sitk.GetImageFromArray(ctnp)
        sitk.WriteImage(volgt,'gt_sub{:02d}'.format(ind)+'.nii.gz')     
         
    
        ### for Discriminator network
        if opt.resType==2:
            matConfLabel, matConfProb, _ = testOneSubjectWith4DInput(matProb, matProb, opt.out_channels_netD, opt.input_sz,opt.output_sz,opt.test_step_sz,netD,opt.netDModelPath,resType=opt.resType, nd = opt.NDim)
        else:
            matConfLabel,_ = testOneSubjectWith4DInput(matProb, matProb, opt.out_channels_netD, opt.input_sz,opt.output_sz,opt.test_step_sz,netD, opt.netDModelPath,resType=opt.resType, nd = opt.NDim)
    
        matConfFGProb = np.squeeze(matConfProb[1,...])
        matConfBGProb = np.squeeze(matConfProb[0,...])
        volProb = sitk.GetImageFromArray(matConfFGProb)
        sitk.WriteImage(volProb,opt.prefixPredictedFN+'_confProb_sub{:02d}'.format(ind)+'.nii.gz')
        volProb = sitk.GetImageFromArray(matConfBGProb)
        sitk.WriteImage(volProb,opt.prefixPredictedFN+'_confProb1_sub{:02d}'.format(ind)+'.nii.gz')        
        volOut = sitk.GetImageFromArray(matConfLabel)
        sitk.WriteImage(volOut,opt.prefixPredictedFN+'_confLabel_sub{:02d}'.format(ind)+'.nii.gz')     
         

if __name__ == '__main__':
#     testGradients()     
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main() 
