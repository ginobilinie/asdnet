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
parser.add_argument("--out_channels", type=int, default=2, help="the output channels (num of classes)?")
# parser.add_argument("--in_slices", type=int, default=3, help="the num of consecutive slices for input unit?")
# parser.add_argument("--out_slices", type=int, default=1, help="the num of consecutive slices for output unit?")
parser.add_argument("--input_sz", type=arg_as_list, default=[16,64,64], help="the input patch size of list")
parser.add_argument("--output_sz", type=arg_as_list, default=[16,64,64], help="the output patch size of list")
parser.add_argument("--test_step_sz", type=arg_as_list, default=[1,16,16], help="the step size at testing one subject")
parser.add_argument("--isSegReg", action="store_true", help="is Seg and Reg?", default=False)
parser.add_argument("--isDiceLoss", action="store_true", help="is Dice Loss used?", default=True)
parser.add_argument("--isSoftmaxLoss", action="store_true", help="is Softmax Loss used?", default=True)
parser.add_argument("--isContourLoss", action="store_true", help="is Contour Loss used?", default=False)
parser.add_argument("--isDeeplySupervised", action="store_true", help="is deeply supervised mechanism used?", default=False)
parser.add_argument("--isHighResolution", action="store_true", help="is high resolution used?", default=True)
parser.add_argument("--isResidualEnhancement", action="store_true", help="is residual learning operation enhanced?", default=False)
parser.add_argument("--isViewExpansion", action="store_true", help="is view expanded?", default=True)
parser.add_argument("--isAdLoss", action="store_true", help="is adversarial loss used?", default=False)
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
parser.add_argument("--prefixModelName", default="/shenlab/lab_stor5/dongnie/challengeData/SegCha_wce_wdice_viewExp_resEnhance_lrdcr_1216_", type=str, help="prefix of the to-be-saved model name")

parser.add_argument("--modelPath", default="/shenlab/lab_stor/dongnie/challengeData/modelFiles/SegCha_3D_onlyGeneDice_viewExp_HR_0427_35000.pt", type=str, help="prefix of the to-be-saved model name")
#parser.add_argument("--modelPath", default="/home/dongnie/Desktop/myPyTorch/pytorch-SRResNet23D/SegCha_wce_wdice_viewExp_resEnhance_lrdcr_1216_100000.pt", type=str, help="prefix of the to-be-saved model name")
#parser.add_argument("--modelPath", default="/shenlab/lab_stor5/dongnie/challengeData/modelFiles/SegCha_3D_wce_wdice_viewExp_resEnhance_lrdcr_0110_140000.pt", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="preSub_model3p5w_onlyGeneDice_viewExp_HR_0427_", type=str, help="prefix of the to-be-saved predicted filename")
parser.add_argument("--how2normalize", type=int, default=4, help="how to normalize the data")
parser.add_argument("--resType", type=int, default=2, help="resType: 0: segmentation map (integer); 1: regression map (continuous); 2: segmentation map + probability map")


def main():
    global opt
    opt = parser.parse_args()
    print opt
    
    path_test = '/home/dongnie/warehouse/pelvicSeg/prostateChallenge/data/'
    path_test = '/shenlab/lab_stor5/dongnie/challengeData/testdata/'
        
    if opt.isSegReg:
        netG = ResSegRegNet(opt.in_channels, opt.out_channels, nd=opt.NDim)
    elif opt.isContourLoss:
        netG = ResSegContourNet(opt.in_channels, opt.out_channels, nd=opt.NDim, isRandomConnection=opt.isResidualEnhancement,isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut,dropoutRate=opt.dropoutRate)
    elif opt.isDeeplySupervised and opt.isHighResolution:
        netG = HRResSegNet_DS(opt.in_channels, opt.out_channels, nd=opt.NDim, isRandomConnection=opt.isResidualEnhancement,isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut,dropoutRate=opt.dropoutRate)
    elif opt.isDeeplySupervised:
        netG = ResSegNet_DS(opt.in_channels, opt.out_channels, nd=opt.NDim, isRandomConnection=opt.isResidualEnhancement,isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut,dropoutRate=opt.dropoutRate)
    elif opt.isHighResolution:
        netG = HRResSegNet(opt.in_channels, opt.out_channels, nd=opt.NDim, isRandomConnection=opt.isResidualEnhancement, isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut,dropoutRate=opt.dropoutRate)
    else:
        netG = ResSegNet(opt.in_channels, opt.out_channels, nd=opt.NDim, isRandomConnection=opt.isResidualEnhancement,isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut,dropoutRate=opt.dropoutRate)
    #netG.apply(weights_init)
    netG = netG.cuda()
    
    checkpoint = torch.load(opt.modelPath)
    netG.load_state_dict(checkpoint["model"])
#     netG.load_state_dict(torch.load(opt.modelPath))
    
    
    ids = [1,2,3,4,6,7,8,10,11,12,13]
    ids = [45,46,47,48,49]
    for ind in ids:
        mr_test_itk=sitk.ReadImage(os.path.join(path_test,'Case%d.nii.gz'%ind))
        ct_test_itk=sitk.ReadImage(os.path.join(path_test,'Case%d_segmentation.nii.gz'%ind))
        
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
        row,col,leng = mrnp.shape
        y1 = int (leng * 0.25)
        y2 = int (leng * 0.75)
        x1 = int (col * 0.25)
        x2 = int(col * 0.75)
#         x1 = 120
#         x2 = 350
#         y1 = 120
#         y2 = 350
        matFA = mrnp[:,y1:y2,x1:x2] #note, matFA and matFAOut same size 
#         matGT = ctnp[:,y1:y2,x1:x2]
#         matFA = mrnp
        #matGT = ctnp
        
        if opt.resType==2:
            matOut, matProb, _ = testOneSubject(matFA,matFA,opt.out_channels,opt.input_sz,opt.output_sz,opt.test_step_sz,netG,opt.modelPath,resType=opt.resType, nd = opt.NDim)
        else:
            matOut,_ = testOneSubject(matFA,matFA,opt.out_channels,opt.input_sz,opt.output_sz,opt.test_step_sz,netG,opt.modelPath,resType=opt.resType, nd = opt.NDim)
                                      
        #matOut,_ = testOneSubject(matFA,matGT,opt.out_channels,opt.input_sz, opt.output_sz, opt.test_step_sz,netG,opt.modelPath, nd = opt.NDim)
        ct_estimated = np.zeros([mrnp.shape[0],mrnp.shape[1],mrnp.shape[2]])
        ct_prob = np.zeros([opt.out_channels, mrnp.shape[0],mrnp.shape[1],mrnp.shape[2]])
        
#         print 'matOut shape: ',matOut.shape
        ct_estimated[:,y1:y2,x1:x2] = matOut
#         ct_estimated = matOut

        ct_prob[:,:,y1:y2,x1:x2] = matProb
#         ct_prob = matProb
        matProb_Bladder = np.squeeze(ct_prob[1,:,:,:])
                
#         volout = sitk.GetImageFromArray(matProb_Bladder)
#         sitk.WriteImage(volout,opt.prefixPredictedFN+'prob_Prostate_sub%02d'%ind+'.nii.gz')  
        
        threshold = 0.9
        inds = np.where(matProb_Bladder>threshold) 
        tmat = np.zeros(matProb_Bladder.shape)
        tmat[inds] = 1
        tmat = denoiseImg_closing(tmat, kernel=np.ones((20,20,20))) 
        tmat = denoiseImg_isolation(tmat, struct=np.ones((3,3,3)))   
        diceBladder = dice(tmat,ctnp,1)       
#         diceBladder = dice(tmat,ctnp,1)
        print 'sub%d'%ind,'dice1 = ',diceBladder
        volout = sitk.GetImageFromArray(tmat)
        sitk.WriteImage(volout,opt.prefixPredictedFN+'threshSeg_sub{:02d}'.format(ind)+'.nii.gz')        
        
        ct_estimated = np.rint(ct_estimated) 
        ct_estimated = denoiseImg_closing(ct_estimated, kernel=np.ones((20,20,20))) 
        ct_estimated = denoiseImg_isolation(ct_estimated, struct=np.ones((3,3,3)))   
        diceBladder = dice(ct_estimated,ctnp,1)
    
#         diceProstate = dice(ct_estimated,ctnp,2)
#         diceRectumm = dice(ct_estimated,ctnp,3)
        
#         print 'pred: ',ct_estimated.dtype, ' shape: ',ct_estimated.shape
#         print 'gt: ',ctnp.dtype,' shape: ',ctnp.shape
#         print 'sub%d'%ind,'dice1 = ',diceBladder,' dice2= ',diceProstate,' dice3= ',diceRectumm
        print 'sub%d'%ind,'dice1 = ',diceBladder
        volout = sitk.GetImageFromArray(ct_estimated)
        sitk.WriteImage(volout,opt.prefixPredictedFN+'sub%02d'%ind+'.nii.gz')
        #volgt = sitk.GetImageFromArray(ctnp)
        #sitk.WriteImage(volgt,'gt_sub{}'.format(ind)+'.nii.gz')  
  
    
    
if __name__ == '__main__':
#     testGradients()     
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main() 