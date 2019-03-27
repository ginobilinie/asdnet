# from __future__ import print_function
import argparse, os
import random
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

import ast

'''
This copy of code is a comprehensive gan (cnn or fcn for discriminator) for segmentation, 
part of the work have been submitted to TNNLS, and the fully convolutional discriminator one is new
I also include the semi/weakly supervised learning into the framework
The Generator is the one I depicted in the TNNLS paper: 
    with enhanced residual module, 
    with view expansion, 
    with spatially-region convolution 
    with deeply supervsied mechanism 
The Discriminator is the one I have described in the TNNLS paper:
    A typical CNN which force the fake (segmented maps) maps' distribution to be close with the real segmented maps
    In addition, I also design a sample attention mechanism which wants to play same role as focal loss, this is not 
    included in the TNNLS paper
    I also implemented a fully convolutional network as the discriminator, in which, I injected a semi-supervised module
    and the weakly-supervised module.
    by Dong Nie
    Jan., 2017 - present
'''
# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--gpuID",default='7',type=str, help="the GPU ID")
parser.add_argument("--isSemiSupervised", action="store_true", help="is the training semi-supervised?", default=False)
parser.add_argument("--NDim", type=int, default=3, help="the dimension of the shape, 1D, 2D or 3D?")
parser.add_argument("--in_channels", type=int, default=1, help="the input channels ?")
parser.add_argument("--out_channels", type=int, default=2, help="the output channels (num of classes)?")
parser.add_argument("--isNetDFullyConv", action="store_true", help="is the netD fully convolutional network?", default=False)
parser.add_argument("--isNetDInputIncludeSource", action="store_true", help="is the input of netD including the source image or not?", default=False)
parser.add_argument("--in_channels_netD", type=int, default=2, help="the input channels for netD?")
parser.add_argument("--out_channels_netD", type=int, default=2, help="the output channels for netD?")
parser.add_argument("--given_weight", type=arg_as_list, default=[1,8], help="the given weight for each organ (bg included)")
parser.add_argument("--given_ids", type=arg_as_list, default=[0,1], help="the given organ id for each organ (bg included)")
parser.add_argument("--input_sz", type=arg_as_list, default=[16,64,64], help="the input patch size of list")
parser.add_argument("--output_sz", type=arg_as_list, default=[16,64,64], help="the output patch size of list")
parser.add_argument("--test_step_sz", type=arg_as_list, default=[2,16,16], help="the step size at testing one subject")
parser.add_argument("--isSegReg", action="store_true", help="is Seg and Reg?", default=False)
parser.add_argument("--isDiceLoss", action="store_true", help="is Dice Loss used?", default=True)
parser.add_argument("--isGeneDiceLoss", action="store_true", help="is Generalized Dice Loss used?", default=True)
parser.add_argument("--isSoftmaxLoss", action="store_true", help="is Softmax Loss used?", default=False)
parser.add_argument("--isContourLoss", action="store_true", help="is Contour Loss used?", default=False)
parser.add_argument("--isDeeplySupervised", action="store_true", help="is deeply supervised mechanism used?", default=True)
parser.add_argument("--isHighResolution", action="store_true", help="is high resolution used?", default=True)
parser.add_argument("--isLongConcatConnection", action="store_true", help="is the long skip connection concatenated?", default=True)
parser.add_argument("--isAttentionConcat", action="store_true", help="do we use attention based concatenation?", default=True)
parser.add_argument("--lambda_ds1", type=float, default=0.4, help="loss coefficient for AD loss. Default=0.1")
parser.add_argument("--lambda_ds2", type=float, default=0.2, help="loss coefficient for AD loss. Default=0.1")
parser.add_argument("--isResidualEnhancement", action="store_true", help="is residual learning operation enhanced?", default=False)
parser.add_argument("--isViewExpansion", action="store_true", help="is view expanded?", default=True)
parser.add_argument("--isAdLoss", action="store_true", help="is adversarial loss used?", default=False)
parser.add_argument("--isSpatialDropOut", action="store_true", help="is spatial dropout used?", default=True)
parser.add_argument("--isFocalLoss", action="store_true", help="is focal loss used?", default=False)
parser.add_argument("--isSampleImportanceFromAd", action="store_true", help="is sample importance from adversarial network used?", default=False)
parser.add_argument("--isConfidenceRegionAware", action="store_true", help="is voxel-wise (region) confidence-aware (like focal loss) loss used?", default=False)
parser.add_argument("--dropoutRate", type=float, default=0.2, help="Spatial Dropout Rate. Default=0.25")
parser.add_argument("--lambdaAD", type=float, default=0.05, help="loss coefficient for AD loss. Default=0.1")
parser.add_argument("--lambdaSEMI", type=float, default=0.5, help="loss coefficient for SEMI loss. Default=0.1")
parser.add_argument("--adImportance", type=float, default=0, help="Sample importance from AD network. Default=0")
parser.add_argument("--th_semi", type=float, default=0.5, help="threshold for semi-supervised learning. Default=0.2")
parser.add_argument("--adaptive_semi", action="store_true", help="adaptive thresholding for the semi-supervised learning?", default=False)

parser.add_argument("--how2normalize", type=int, default=4, help="how to normalize the data")
parser.add_argument("--batchSize", type=int, default=3, help="training batch size")
parser.add_argument("--iter_size", type=int, default=3, help="number of iterations(actually, it is batches) to update the network parameters")
parser.add_argument("--numofIters", type=int, default=100000, help="number of iterations to train for")
parser.add_argument("--showTrainLossEvery", type=int, default=100, help="number of iterations to show train loss")
parser.add_argument("--saveModelEvery", type=int, default=5000, help="number of iterations to save the model")
parser.add_argument("--showTestPerformanceEvery", type=int, default=1000, help="number of iterations to show test performance")
parser.add_argument("--lr", type=float, default=2e-3, help="Learning Rate. Default=1e-4")
parser.add_argument("--lr_netD", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--decLREvery", type=int, default=10000, help="Sets the learning rate to the initial LR decayed by momentum every n iterations, Default: n=40000")
parser.add_argument("--cuda", action="store_true", help="Use cuda?", default=True)
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--resume_netD", default="", type=str, help="Path to checkpoint of discrimnator (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
#parser.add_argument("--prefixModelName", default="/home/dongnie/Desktop/myPyTorch/pytorch-SRResNet23D/SegCha_3D_wce_wdice_viewExp_resEnhance_lrdcr_fullAd_0116_", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixModelName", default="/shenlab/lab_stor/dongnie/challengeData/modelFiles/SegCha_3D_onlyGenDice_lr2e3_lrdce_viewExp_DS_Dp_HR_SE_UNet_0805_", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="preSub45_cha_3D_onlyGenDice_lr2e3_lrdce_viewExp_DS_Dp_SE_UNet_HR_0805_", type=str, help="prefix of the to-be-saved predicted filename")
parser.add_argument("--isTestonAttentionRegion", action="store_true", help="Test on the attention region?", default=True)

parser.add_argument("--test_input_file_name",default='Case45.nii.gz',type=str, help="the input file name for testing subject")
parser.add_argument("--test_gt_file_name",default='Case45_segmentation.nii.gz',type=str, help="the ground-truth file name for testing subject")
parser.add_argument("--path_test",default='/shenlab/lab_stor5/dongnie/challengeData/data',type=str, help="the path for the testing nii.gz files")
parser.add_argument("--path_patients_h5",default='/shenlab/lab_stor5/dongnie/challengeData/pelvicSegRegContourBatchH5',type=str, help="the path for the training hdf5 files")
parser.add_argument("--path_patients_h5_test",default='/shenlab/lab_stor5/dongnie/challengeData/pelvicSegRegContourH5Test',type=str, help="the path for the testing hdf5 files")
parser.add_argument("--path_patients_unlabeled_h5",default='/shenlab/lab_stor5/dongnie/challengeData/pelvicSegUnlabeledH5',type=str, help="the path for the training unlabeled hdf5 files")


#parser.add_argument("--path_test",default='/home/dongnie/warehouse/pelvicSeg/prostateChallenge/data',type=str, help="the path for the testing nii.gz files")
#parser.add_argument("--path_patients_h5",default='/home/dongnie/warehouse/pelvicSeg/prostateChallenge/pelvic3DSegRegContourBatchH5',type=str, help="the path for the training hdf5 files")
#parser.add_argument("--path_patients_unlabeled_h5",default='/home/dongnie/warehouse/pelvicSeg/prostateChallenge/pelvic3DSegRegContourBatchH5',type=str, help="the path for the training unlabeled hdf5 files")
#parser.add_argument("--path_patients_h5_test",default='/home/dongnie/warehouse/pelvicSeg/prostateChallenge/pelvic3DSegRegContourH5Test',type=str, help="the path for the testing hdf5 files")


global opt, model, running_loss, start, criterion_dice, criterion_CEND, data_generator_test, path_test
opt = parser.parse_args()
running_loss = 0
def main():    

########################################configs####################################
    #opt = parser.parse_args()
    print opt
    print 'test my list, opt.input_sz: ',opt.input_sz
    if opt.isSemiSupervised:
        trainSemiSupervisedNet()
    else:
        trainSupervisedNet()

'''
    train supervised network
'''
def trainSupervisedNet():
    given_weight = torch.FloatTensor(opt.given_weight) #note, weights for each organ
    given_ids = torch.FloatTensor(opt.given_ids)
    given_weight = given_weight.cuda()
    given_ids = given_ids.cuda()

    path_test = opt.path_test
    path_patients_h5 = opt.path_patients_h5
    path_patients_h5_test = opt.path_patients_h5_test

########################################configs####################################
#     running_loss = 0
    ## step.1 prepare data flow
    global data_generator_test,path_test,criterion_CEND,criterion_dice
    if opt.NDim == 3:
        data_generator = Generator_3D_patches(path_patients_h5,opt.batchSize,inputKey='dataMR',outputKey='dataSeg')
        data_generator_test = Generator_3D_patches(path_patients_h5_test,opt.batchSize,inputKey='dataMR',outputKey='dataSeg')
    else:
        data_generator_test = Generator_2D_slices(path_patients_h5_test,opt.batchSize,inputKey='dataMR2D',outputKey='dataSeg2D')
        if opt.isSegReg:
            data_generator = Generator_2D_slices_variousKeys(path_patients_h5,opt.batchSize,inputKey='dataMR2D',outputKey='dataSeg2D',regKey1='dataBladder2D',regKey2='dataProstate2D',regKey3='dataRectum2D')
        elif opt.isContourLoss:
            data_generator = Generator_2D_slicesV1(path_patients_h5,opt.batchSize,inputKey='dataMR2D',segKey='dataSeg2D',contourKey='dataContour2D')
        else:
            data_generator = Generator_2D_slices(path_patients_h5,opt.batchSize,inputKey='dataMR2D',outputKey='dataSeg2D')
#     inputs=Variable(torch.randn(1000,1,32,32)) #here should be tensor instead of variable
#     targets=Variable(torch.randn(1000,10,1,1)) #here should be tensor instead of variable
#     trainset=data_utils.TensorDataset(inputs, targets)
#     trainloader = data_utils.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#     inputs=torch.randn(1000,1,32,32)
#     targets=torch.LongTensor(1000)

    ## step.2 prepare network architecture
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
    elif opt.isLongConcatConnection and opt.isDeeplySupervised and opt.isHighResolution:
        netG = HRResUNet_DS(opt.in_channels, opt.out_channels, isRandomConnection=opt.isResidualEnhancement, isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut, dropoutRate=opt.dropoutRate, TModule='HR', FModule=opt.isAttentionConcat, nd=opt.NDim)
    elif opt.isLongConcatConnection and opt.isHighResolution:
        netG = UNet(opt.in_channels, opt.out_channels, TModule='HR', FModule=opt.isAttentionConcat, nd=opt.NDim)
    elif opt.isLongConcatConnection:
        netG = UNet(opt.in_channels, opt.out_channels, TModule=None, FModule=opt.isAttentionConcat, nd=opt.NDim)
    else:
        netG = ResSegNet(opt.in_channels, opt.out_channels, nd=opt.NDim, isRandomConnection=opt.isResidualEnhancement,isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut,dropoutRate=opt.dropoutRate)
    #netG.apply(weights_init)
    netG = netG.cuda()
    
    if opt.isAdLoss:
        if opt.isNetDFullyConv:
            netD = Discriminator_my23DLRResFCN(opt.in_channels_netD, opt.out_channels_netD, nd=opt.NDim)
        else:
            netD = Discriminator(opt.in_channels_netD, nd=opt.NDim)
        netD.apply(weights_init)
        netD = netD.cuda()
        optimizerD =optim.Adam(netD.parameters(),lr=opt.lr_netD)
    params = list(netG.parameters())
    print('len of params is ')
    print(len(params))
    print('size of params is ')
    print(params[0].size())
    optimizerG =optim.Adam(netG.parameters(),lr=opt.lr)

    
    ## step.3 prepare criterion (loss function)
#     criterion_NLL2D = nn.NLLLoss2d(weight=given_weight)
    if opt.NDim==2:
        criterion_CEND = CrossEntropy2d(weight=given_weight)
    elif opt.NDim==3: 
        criterion_CEND = CrossEntropy3d(weight=given_weight)
        criterion_WCEND = WeightedCrossEntropy3d(weight=given_weight)
        criterion_BCEND = CrossEntropy3d()
    
    criterion_BCE2D = CrossEntropy2d()#for contours
#     criterion_dice = DiceLoss4Organs(organIDs=[1,2,3], organWeights=[1,1,1])
#     criterion_dice = WeightedDiceLoss4Organs()
    criterion_dice = myWeightedDiceLoss4Organs(organIDs=given_ids, organWeights = given_weight)
    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    criterion_dice = criterion_dice.cuda()
    criterion_CEND = criterion_CEND.cuda()
    criterion_BCEND = criterion_BCEND.cuda()
    criterion_BCE2D = criterion_BCE2D.cuda()
    criterion_WCEND = criterion_WCEND.cuda()
    
    if opt.isSegReg:
        criterion_MSE = nn.MSELoss()
        criterion_MSE = criterion_MSE.cuda()

    if opt.isFocalLoss:
        criterion_focal = myFocalLoss(4, alpha=given_weight, gamma=2)  
        criterion_focal = criterion_focal.cuda()
#     softmax2d = nn.Softmax2d()
    if opt.isGeneDiceLoss:
        criterion_dice = GeneralizedDiceLoss4Organs(organIDs=given_ids)    
        criterion_dice = criterion_dice.cuda()
    elif opt.isDiceLoss:
        criterion_dice = myWeightedDiceLoss4Organs(organIDs=given_ids, organWeights = given_weight)
        criterion_dice = criterion_dice.cuda()
        
    ## step.4 check if we should resume training using the existing models
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            netG.load_state_dict(checkpoint["model"])
            if opt.isAdLoss:
            	checkpoint = torch.load(opt.resume_netD)
                netD.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
#     running_loss = 0.0
    global start
    start = time.time()
    for iter in range(opt.start_epoch, opt.numofIters+1):
        ## step.5.1 prepare training data (input, labels)
        #we should consider different data to train
        if opt.isAdLoss:
            netD.zero_grad()
        netG.zero_grad()
        lossG_G = Variable(torch.FloatTensor([0]).cuda()) 
        loss_dice = Variable(torch.FloatTensor([0]).cuda()) 
        lossG_D = Variable(torch.FloatTensor([0]).cuda()) 
        lossG_focal = Variable(torch.FloatTensor([0]).cuda()) 
        lossG_contour = Variable(torch.FloatTensor([0]).cuda()) 
        lossG_Reg = Variable(torch.FloatTensor([0]).cuda()) 
        lossD = Variable(torch.FloatTensor([0]).cuda()) 
        for m in range(0,opt.iter_size):
            if opt.isSegReg:
                inputs,labels, regGT1, regGT2, regGT3 = data_generator.next()
            elif opt.isContourLoss:
                inputs,labels,contours = data_generator.next()
            else:
                inputs,labels = data_generator.next()
                #print inputs.size,labels.size
    
            labels = np.squeeze(labels,axis=1)
            labels = labels.astype(int)
            
            if opt.isContourLoss:
                contours = np.squeeze(contours,axis=1)
                contours = contours.astype(int)
                contours = torch.from_numpy(contours)
                contours = contours.cuda()
                contours = Variable(contours)
            
            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels)
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            #wrap them into Variable
            inputs,labels = Variable(inputs),Variable(labels)
            
            
            ## step.5.2 prepare the training data for adversarial training
            if opt.isAdLoss:
                if opt.isSegReg:
                    outputG, outputReg1, outputReg2, outputReg3 = netG(inputs)
                elif opt.isContourLoss:    
                    outputG,_ = netG(inputs)
                elif opt.isDeeplySupervised:
                    outputG, outputG_path1, outputG_path2 = netG(inputs)
                else:
                    outputG = netG(inputs)
                #get the probability map of outputG    
#                 if opt.NDim==2:
#                     outputG = softmax2d(outputG) #batch
#                 elif opt.NDim==3:
                outputG = F.softmax(outputG, dim=1)
        
                #outputG = outputG.data.max(1)[1]
                #outputG = torch.squeeze(outputG) #[N,C,W,H]
                #labels = labels.unsqueeze(1) #expand the 1st dim
                #one hot encode for the gt labels
                #ohlabels = OneHotEncode(nclass=2,nd=3)(labels.data.cpu()) #NxWxHxD->NxCxWxHxD
                ohlabels = OneHotEncode(nclass=opt.out_channels,nd=opt.NDim)(labels.data.cpu()) #NxWxHxD->NxCxWxHxD
                # print 'outputG: ',outputG.size(),'labels: ',ohlabels.size()
                outputR = Variable(ohlabels.type(torch.FloatTensor).cuda()) #output_Real
                outputG = outputG.type(torch.FloatTensor).cuda()
                
                if opt.isNetDInputIncludeSource:
                    inputReal_netD = torch.cat((inputs,outputR),dim=1)
                    outputD_real = netD(inputReal_netD)
                    inputFake_netD = torch.cat((inputs,outputG),dim=1)
                    outputD_fake = netD(inputFake_netD)
                else:
                    outputD_real = netD(outputR)
                    outputD_fake = netD(outputG)
                    
                
                ## step.5.2 update D network: maximize log(D(x)) + log(1 - D(G(z)))
                
                batch_size = inputs.size(0)
                output_shape = outputD_real.shape
        
                if opt.isNetDFullyConv: #if netD is FCN based
                    #train with real data
                    if opt.NDim==3:
                        real_label = torch.ones(output_shape[0],output_shape[2],output_shape[3],output_shape[4]).long()
                        fake_label = torch.zeros(output_shape[0],output_shape[2],output_shape[3],output_shape[4]).long()
                    else:
                        real_label = torch.ones(output_shape[0],output_shape[2],output_shape[3]).long()
                        fake_label = torch.zeros(output_shape[0],output_shape[2],output_shape[3]).long()
                    real_label = Variable(real_label.cuda())
                    loss_real = criterion_BCEND(outputD_real,real_label)
                    loss_real.backward()
                    #train with fake data
                    fake_label = Variable(fake_label.cuda())  
                    loss_fake = criterion_BCEND(outputD_fake,fake_label)
                    loss_fake.backward()      
                else: #if netD is CNN based
                    #train with real data
                    real_label = torch.ones(batch_size,1)
                    real_label = Variable(real_label.cuda())
                    loss_real = criterion(outputD_real,real_label)
                    loss_real.backward()
                    #train with fake data
                    fake_label=torch.zeros(batch_size,1)
                    fake_label = Variable(fake_label.cuda()) 
                    loss_fake = criterion(outputD_fake,fake_label)
                    loss_fake.backward()
                
                lossD = lossD + (loss_real + loss_fake)
        
            
            ##step.5.3 update G network: minimize the L1/L2 loss, maximize the D(G(x))
            
            #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
            #view of equation (note the max and min difference for generator and discriminator)
            if opt.isAdLoss:
                if opt.isSegReg:
                    outputG, outputReg1, outputReg2, outputReg3 = netG(inputs)
                elif opt.isContourLoss:
                    outputG,_ = netG(inputs)
                elif opt.isDeeplySupervised:
                    outputG, outputG_path1, outputG_path2 = netG(inputs)
                else:
                    outputG = netG(inputs)
                #obtain the prediction probability maps    
#                 if opt.NDim==2:
#                     outputG = softmax2d(outputG) #batach
#                 elif opt.NDim==3:
                outputG = F.softmax(outputG, dim=1)
        
    #             outputG = outputG.data.max(1)[1]
                outputG = outputG.type(torch.FloatTensor).cuda() #we directly use the probabilities: NxCxWxHxD
                
                if opt.isNetDInputIncludeSource:
                    input_netD = torch.cat((inputs,outputG),dim=1)
                    outputD = netD(input_netD)
                else:
                    outputD = netD(outputG)
                #obtain the prediction probability maps    
                outputD = F.softmax(outputD, dim=1)
                 
                #compute the non-zero item probability, here we already use data, so it is not variable any more        
                averProbTensor = (1 - outputD.data[0].cpu()) 
                
                if opt.isNetDFullyConv and opt.isConfidenceRegionAware:
                    confRegion = computeVoxelAttentionWeight(averProbTensor)
                    confRegion = Variable(confRegion.cuda(),requires_grad=False)

                if opt.isSampleImportanceFromAd:
                    averProb = averProbTensor.mean()
                    adImportance = computeSampleAttentionWeight(averProb)
                if opt.isNetDFullyConv:       
                    lossG_D = lossG_D + opt.lambdaAD * criterion_BCEND(outputD, real_label) #note, for generator, the label for outputG is real
                else:
                    lossG_D = lossG_D + opt.lambdaAD * criterion(outputD, real_label) #note, for generator, the label for outputG is real
                lossG_D.backward(retain_graph=True)
            
            if opt.isSegReg:
                outputG, outputReg1, outputReg2, outputReg3 = netG(inputs)
            elif opt.isContourLoss:
                outputG,outputContour = netG(inputs)
            elif opt.isDeeplySupervised:
                outputG, outputG_path1, outputG_path2 = netG(inputs)
            else:
                outputG = netG(inputs) #here I am not sure whether we should use twice or not
            
            if opt.isSoftmaxLoss:
                if opt.isNetDFullyConv and opt.isConfidenceRegionAware:
                    lossG_G_main = criterion_WCEND(outputG,torch.squeeze(labels,dim=1),confRegion) 
                elif opt.isSampleImportanceFromAd:
                    lossG_G_main = (1+adImportance) * criterion_CEND(outputG,torch.squeeze(labels,dim=1)) 
                else:
                    lossG_G_main = criterion_CEND(outputG,torch.squeeze(labels,dim=1)) 
                if opt.isDeeplySupervised:
                    lossG_G_path1 = opt.lambda_ds1*criterion_CEND(outputG_path1,torch.squeeze(labels,dim=1)) 
                    lossG_G_path2 = opt.lambda_ds2*criterion_CEND(outputG_path2,torch.squeeze(labels,dim=1)) 
                    lossG_G = lossG_G + (lossG_G_main + lossG_G_path1 + lossG_G_path2)
                else:
                    lossG_G = lossG_G + lossG_G_main
                lossG_G.backward(retain_graph=True) #compute gradients
            
            if opt.isDiceLoss:
    #             criterion_dice = myWeightedDiceLoss4Organs(organIDs=[0,1,2,3], organWeights=[1,4,8,6])
                if opt.isSampleImportanceFromAd:
                    loss_dice_main = (1+adImportance) * criterion_dice(outputG,torch.squeeze(labels,dim=1))
                else:
                    loss_dice_main = criterion_dice(outputG,torch.squeeze(labels,dim=1))
                if opt.isDeeplySupervised:
                    loss_dice_path1 = opt.lambda_ds1*criterion_dice(outputG_path1,torch.squeeze(labels,dim=1)) 
                    loss_dice_path2 = opt.lambda_ds2*criterion_dice(outputG_path2,torch.squeeze(labels,dim=1)) 
                    loss_dice = loss_dice + (loss_dice_main + loss_dice_path1 + loss_dice_path2)
                else:
                    loss_dice = loss_dice + loss_dice_main
    #             loss_dice = myDiceLoss4Organs(outputG,torch.squeeze(labels)) #succeed
    #             loss_dice.backward(retain_graph=True) #compute gradients for dice loss
                loss_dice.backward(retain_graph=True) #compute gradients for dice loss
            
            if opt.isContourLoss:
                lossG_contour = lossG_contour + criterion_BCE2D(outputContour,contours)
                lossG_contour.backward(retain_graph=True)
    
            if opt.isSegReg:
                lossG_Reg1 = criterion_MSE(outputReg1, regGT1)
                lossG_Reg2 = criterion_MSE(outputReg2, regGT2)
                lossG_Reg3 = criterion_MSE(outputReg3, regGT3)
                lossG_Reg = lossG_Reg + (lossG_Reg1 + lossG_Reg2 + lossG_Reg3)
                lossG_Reg.backward()
    
            if opt.isFocalLoss:
                lossG_focal = lossG_focal + criterion_focal(outputG,torch.squeeze(labels,dim=1))
                lossG_focal.backward(retain_graph=True) #compute gradients        

        #update network parameters after iter_size weight computation
        if opt.isAdLoss:
			optimizerD.step()
        optimizerG.step() 
        #empty memory
        #del outputD
        #del outputG
        del inputs
        del labels
        
        if opt.isAdLoss:
            showTestStatistics(netG, lossG_G/opt.iter_size, loss_dice/opt.iter_size, lossG_D/opt.iter_size, lossG_focal/opt.iter_size, lossG_contour/opt.iter_size, lossG_Reg/opt.iter_size, lossD/opt.iter_size, iter, netD=netD)
        else:
            showTestStatistics(netG, lossG_G/opt.iter_size, loss_dice/opt.iter_size, lossG_D/opt.iter_size, lossG_focal/opt.iter_size, lossG_contour/opt.iter_size, lossG_Reg/opt.iter_size, lossD/opt.iter_size, iter)
        #showTestStatistics(netG, lossG_G, loss_dice, lossG_D, lossG_focal, lossG_contour, lossG_Reg, lossD, iter, netD=netD)
        
        if iter%opt.decLREvery==0 and iter>0:
            opt.lr = opt.lr*0.5
            adjust_learning_rate(optimizerG, opt.lr)
            print 'now the learning rate is {}'.format(opt.lr)
    print('Finished Training')
    

'''
    A function to show the test statistics during training
    a).show training loss
    b).show validation loss
    c).show accuracy performance on test subjects
'''
def showTestStatistics(netG, lossG_G, loss_dice, lossG_D, lossG_focal, lossG_contour, lossG_Reg, lossD, iter, labeled=True, netD=None):
    dsc = -1
    if labeled:
        if opt.isDiceLoss and opt.isSoftmaxLoss and opt.isAdLoss and opt.isSegReg and opt.isFocalLoss:
            lossG = opt.lambdaAD * lossG_D + lossG_G+loss_dice.data[0] + lossG_Reg + lossG_focal
        elif opt.isDiceLoss and opt.isSoftmaxLoss and opt.isSegReg and opt.isFocalLoss:
            lossG = lossG_G+loss_dice.data[0] + lossG_Reg + lossG_focal
        elif opt.isDiceLoss and opt.isFocalLoss and opt.isAdLoss and opt.isSegReg:
            lossG = opt.lambdaAD * lossG_D + lossG_focal + loss_dice.data[0] + lossG_Reg
        elif opt.isDiceLoss and opt.isFocalLoss and opt.isSegReg:
            lossG = lossG_focal + loss_dice.data[0] + lossG_Reg
        elif opt.isDiceLoss and opt.isSoftmaxLoss and opt.isAdLoss and opt.isSegReg:
            lossG = opt.lambdaAD * lossG_D + lossG_G+loss_dice.data[0] + lossG_Reg
        elif opt.isDiceLoss and opt.isSoftmaxLoss and opt.isSegReg:
            lossG =  lossG_G+loss_dice.data[0] + lossG_Reg
        elif opt.isSoftmaxLoss and opt.isAdLoss and opt.isSegReg:
            lossG = opt.lambdaAD * lossG_D + lossG_G + lossG_Reg
        elif opt.isSoftmaxLoss and opt.isSegReg:
            lossG = lossG_G + lossG_Reg
        elif opt.isDiceLoss and opt.isAdLoss and opt.isSegReg:
            lossG = opt.lambdaAD * lossG_D + loss_dice.data[0] + lossG_Reg    
        elif opt.isDiceLoss and opt.isSegReg:
            lossG = loss_dice.data[0] + lossG_Reg    
        elif opt.isDiceLoss and opt.isSoftmaxLoss and opt.isAdLoss:
            lossG = opt.lambdaAD * lossG_D + lossG_G + loss_dice.data[0]
        elif opt.isDiceLoss and opt.isSoftmaxLoss:
            lossG = lossG_G + loss_dice.data[0]
        elif opt.isDiceLoss and opt.isFocalLoss and opt.isAdLoss:
            lossG = opt.lambdaAD * lossG_D + lossG_focal + loss_dice.data[0]
        elif opt.isDiceLoss and opt.isFocalLoss:
            lossG = lossG_focal + loss_dice.data[0]      
        elif opt.isSoftmaxLoss and opt.isAdLoss:
            lossG = opt.lambdaAD * lossG_D + lossG_G
        elif opt.isSoftmaxLoss:
            lossG = lossG_G
        elif opt.isFocalLoss and opt.isAdLoss:
            lossG = opt.lambdaAD * lossG_D + lossG_focal  
        elif opt.isFocalLoss:
            lossG = lossG_focal      
        elif opt.isDiceLoss and opt.isAdLoss:
            lossG = opt.lambdaAD * lossG_D + loss_dice.data[0]
        elif opt.isDiceLoss:
            lossG = loss_dice
    else:
        if opt.isAdLoss:
            lossG = lossG_D
    #print('loss for generator is %f'%lossG.data[0])
    #print statistics
    global running_loss, start
    running_loss = running_loss + lossG.data[0]
#         print 'running_loss is ',running_loss,' type: ',type(running_loss)
    
#         print type(outputD_fake.cpu().data[0].numpy())
    
    if iter%opt.showTrainLossEvery==0: #print every 2000 mini-batches
        print '************************************************'
        print 'time now is: ' + time.asctime(time.localtime(time.time()))
        #if opt.isAdLoss:
        #    print 'the outputD_real for iter {}'.format(iter), ' is ',outputD_real.cpu().data[0].numpy()[0]
        #    print 'the outputD_fake for iter {}'.format(iter), ' is ',outputD_fake.cpu().data[0].numpy()[0]
#             print 'running loss is ',running_loss
        print 'average running loss for generator between iter [%d, %d] is: %.3f'%(iter - 100 + 1,iter,running_loss/100)
        if opt.isAdLoss:
            print 'loss for discriminator at iter ',iter, ' is %f'%lossD.data[0]
        print 'total loss for generator at iter ',iter, ' is %f'%lossG.data[0]
        if opt.isDiceLoss and opt.isSoftmaxLoss and opt.isAdLoss and opt.isSegReg:
            print 'lossG_D, lossG_G and loss_dice loss_Reg are %.2f, %.2f and %.2f respectively.'%(lossG_D.data[0], lossG_G.data[0], loss_dice.data[0], lossG_Reg.data[0])
        elif opt.isDiceLoss and opt.isSoftmaxLoss and opt.isAdLoss:
            print 'lossG_D, lossG_G and loss_dice are %.2f, %.2f and %.2f respectively.'%(lossG_D.data[0], lossG_G.data[0], loss_dice.data[0])
        elif opt.isDiceLoss and opt.isSoftmaxLoss:
            print 'lossG_G and loss_dice are %.2f and %.2f respectively.'%(lossG_G.data[0], loss_dice.data[0])
        elif opt.isDiceLoss and opt.isFocalLoss and opt.isAdLoss:
            print 'lossG_D, lossG_focal and loss_dice are %.2f, %.2f and %.2f respectively.'%(lossG_D.data[0], lossG_focal.data[0], loss_dice.data[0])    
        elif opt.isSoftmaxLoss and opt.isAdLoss:
            print 'lossG_D and lossG_G are %.2f and %.2f respectively.'%(lossG_D.data[0], lossG_G.data[0])
        elif opt.isFocalLoss and opt.isAdLoss:
            print 'lossG_D and lossG_focal are %.2f and %.2f respectively.'%(lossG_D.data[0], lossG_focal.data[0])    
        elif opt.isDiceLoss and opt.isAdLoss:
            print 'lossG_D and loss_dice are %.2f and %.2f respectively.'%(lossG_D.data[0], loss_dice.data[0])
        
        if opt.isContourLoss:
            print 'lossG_contour is {}'.format(lossG_contour.data[0])

        print 'cost time for iter [%d, %d] is %.2f'%(iter - 100 + 1,iter, time.time()-start)
        print '************************************************'
        running_loss = 0.0
       
        start = time.time()
    if iter%opt.saveModelEvery==0: #save the model
        state={
            'epoch': iter + 1,
            #'state_dict': netG.state_dict(),
            'model': netG.state_dict(),
        }
        torch.save(state,opt.prefixModelName+'%d.pt'%iter)
        if opt.isAdLoss:
            torch.save(netD.state_dict(), opt.prefixModelName+'netD_%d.pt'%iter)
        print 'save model: '+opt.prefixModelName+'%d.pt'%iter
    
    if iter%opt.showTestPerformanceEvery==0: #test one subject  
        # to test on the validation dataset in the format of h5 
        inputs,labels = data_generator_test.next()
        labels = np.squeeze(labels,axis=1)
        labels = labels.astype(int)
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        inputs = inputs.cuda()
        labels = labels.cuda()
        inputs,labels = Variable(inputs),Variable(labels)
        if opt.isSegReg:
            outputG, outputReg1, outputReg2, outputReg3 = netG(inputs)
        elif opt.isContourLoss: 
            outputG,_ = netG(inputs)
        elif opt.isDeeplySupervised:
            outputG, outputG_path1, outputG_path2 = netG(inputs)
        else:
            outputG = netG(inputs) #here I am not sure whether we should use twice or not
        
        lossG_G = criterion_CEND(outputG,torch.squeeze(labels,dim=1))
        loss_dice = criterion_dice(outputG,torch.squeeze(labels,dim=1))
        del outputG
        print '.......come to validation stage: iter {}'.format(iter),'........'
        print 'lossG_G and loss_dice are %.2f and %.2f respectively.'%(lossG_G.data[0], loss_dice.data[0])
        
        ####release all the unoccupied memory####
        torch.cuda.empty_cache()

        mr_test_itk=sitk.ReadImage(os.path.join(path_test,opt.test_input_file_name))
        ct_test_itk=sitk.ReadImage(os.path.join(path_test,opt.test_gt_file_name))
        
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
        if opt.isTestonAttentionRegion:
            # the attention regions
            y1 = int (leng * 0.25)
            y2 = int (leng * 0.75)
            x1 = int (col * 0.25)
            x2 = int(col * 0.75)
            matFA = mrnp[:,y1:y2,x1:x2] #note, matFA and matFAOut same size
            matGT = ctnp[:,y1:y2,x1:x2]
        else:
            matFA = mrnp
            matGT = ctnp

#                 volFA = sitk.GetImageFromArray(matFA)
#                 sitk.WriteImage(volFA,'volFA'+'.nii.gz')
#                 volGT = sitk.GetImageFromArray(matGT)
#                 sitk.WriteImage(volGT,'volGT'+'.nii.gz')
        
        matOut,_ = testOneSubject(matFA,matGT, opt.out_channels, opt.input_sz, opt.output_sz, opt.test_step_sz, netG,opt.prefixModelName+'%d.pt'%iter, nd=opt.NDim)
        ct_estimated = np.zeros([ctnp.shape[0],ctnp.shape[1],ctnp.shape[2]])
        print 'matOut shape: ',matOut.shape
        if opt.isTestonAttentionRegion:
            ct_estimated[:, y1:y2, x1:x2] = matOut
        else:
            ct_estimated = matOut

        ct_estimated = np.rint(ct_estimated) 
        ct_estimated = denoiseImg_closing(ct_estimated, kernel=np.ones((20,20,20)))   
        ct_estimated = denoiseImg_isolation(ct_estimated, struct=np.ones((3,3,3)))
        diceBladder = dice(ct_estimated,ctnp,1)
#             diceProstate = dice(ct_estimated,ctnp,2)
#             diceRectumm = dice(ct_estimated,ctnp,3)
        
        print 'pred: ',ct_estimated.dtype, ' shape: ',ct_estimated.shape
        print 'gt: ',ctnp.dtype,' shape: ',ct_estimated.shape
        #print 'dice1 = ',diceBladder,' dice2= ',diceProstate,' dice3= ',diceRectumm
        print 'dice1 = ',diceBladder

        volout = sitk.GetImageFromArray(ct_estimated)
        sitk.WriteImage(volout,opt.prefixPredictedFN+'{}'.format(iter)+'.nii.gz')    
#             netG.save_state_dict('Segmentor_model_%d.pt'%iter)
#             netD.save_state_dic('Discriminator_model_%d.pt'%iter)
        dsc = diceBladder
    return dsc

'''
    semi-supervised network
'''
def trainSemiSupervisedNet():
    given_weight = torch.FloatTensor(opt.given_weight) #note, weights for each organ
    given_ids = torch.FloatTensor(opt.given_ids)
    given_weight = given_weight.cuda()
    given_ids = given_ids.cuda()

    path_test = opt.path_test
    path_patients_h5 = opt.path_patients_h5
    path_patients_unlabeled_h5 = opt.path_patients_unlabeled_h5
    path_patients_h5_test = opt.path_patients_h5_test

########################################configs####################################

    global data_generator_test,path_test,criterion_CEND,criterion_dice
    
    ## step.1 prepare data flow
    if opt.NDim == 3:
        data_generator = Generator_3D_patches(path_patients_h5,opt.batchSize,inputKey='dataMR',outputKey='dataSeg')
        data_generator_test = Generator_3D_patches(path_patients_h5_test,opt.batchSize,inputKey='dataMR',outputKey='dataSeg')

    else:
        data_generator_test = Generator_2D_slices(path_patients_h5_test,opt.batchSize,inputKey='dataMR2D',outputKey='dataSeg2D')
        if opt.isSegReg:
            data_generator = Generator_2D_slices_variousKeys(path_patients_h5,opt.batchSize,inputKey='dataMR2D',outputKey='dataSeg2D',regKey1='dataBladder2D',regKey2='dataProstate2D',regKey3='dataRectum2D')
        elif opt.isContourLoss:
            data_generator = Generator_2D_slicesV1(path_patients_h5,opt.batchSize,inputKey='dataMR2D',segKey='dataSeg2D',contourKey='dataContour2D')
        else:
            data_generator = Generator_2D_slices(path_patients_h5,opt.batchSize,inputKey='dataMR2D',outputKey='dataSeg2D')
    if opt.isSemiSupervised:
        data_generator_unlabeled = Generator_3D_patches_unlabeled(path_patients_unlabeled_h5,opt.batchSize,inputKey='dataMR')
#     inputs=Variable(torch.randn(1000,1,32,32)) #here should be tensor instead of variable
#     targets=Variable(torch.randn(1000,10,1,1)) #here should be tensor instead of variable
#     trainset=data_utils.TensorDataset(inputs, targets)
#     trainloader = data_utils.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#     inputs=torch.randn(1000,1,32,32)
#     targets=torch.LongTensor(1000)

    ## step.2 prepare network architecture
    if opt.isSegReg:
        negG = ResSegRegNet(opt.in_channels, opt.out_channels, nd=opt.NDim)
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
    netG.cuda()
    
    if opt.isAdLoss:
        if opt.isNetDFullyConv:
            netD = Discriminator_my23DLRResFCN(opt.in_channels_netD, opt.out_channels_netD, nd=opt.NDim)
        else:
            netD = Discriminator(opt.in_channels_netD, opt.out_channels_netD, nd=opt.NDim)
        netD.apply(weights_init)
        netD.cuda()
        optimizerD =optim.Adam(netD.parameters(),lr=opt.lr_netD)
    params = list(netG.parameters())
    print('len of params is ')
    print(len(params))
    print('size of params is ')
    print(params[0].size())
    optimizerG =optim.Adam(netG.parameters(),lr=opt.lr)

    
    ## step.3 prepare criterion (loss function)
#     criterion_NLL2D = nn.NLLLoss2d(weight=given_weight)
    if opt.NDim==2:
        criterion_CEND = CrossEntropy2d(weight=given_weight)
    elif opt.NDim==3: 
        criterion_CEND = CrossEntropy3d(weight=given_weight)
        criterion_BCEND = CrossEntropy3d()
        criterion_WCEND = WeightedCrossEntropy3d(weight=given_weight)
    
    criterion_BCE2D = CrossEntropy2d()#for contours
#     criterion_dice = DiceLoss4Organs(organIDs=[1,2,3], organWeights=[1,1,1])
#     criterion_dice = WeightedDiceLoss4Organs()

    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    criterion_CEND = criterion_CEND.cuda()
    criterion_WCEND = criterion_WCEND.cuda()
    criterion_BCEND = criterion_BCEND.cuda()
    criterion_BCE2D = criterion_BCE2D.cuda()
    
    if opt.isSegReg:
        criterion_MSE = nn.MSELoss()
        criterion_MSE = criterion_MSE.cuda()   

    if opt.isGeneDiceLoss:
        criterion_dice = GeneralizedDiceLoss4Organs(organIDs=given_ids)    
        criterion_dice = criterion_dice.cuda()
    elif opt.isDiceLoss:
        criterion_dice = myWeightedDiceLoss4Organs(organIDs=given_ids, organWeights = given_weight)
        criterion_dice = criterion_dice.cuda()

    if opt.isFocalLoss:
        criterion_focal = myFocalLoss(4, alpha=given_weight, gamma=2)  
        criterion_focal = criterion_focal.cuda()
#     softmax2d = nn.Softmax2d()
    
    
    ## step.4 check if we should resume training using the existing models
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            netG.load_state_dict(checkpoint["model"])
            if opt.isAdLoss:
                netD.load_state_dict(torch.load(opt.resume_netD))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
#     running_loss = 0.0
    global start
    start = time.time()
    currDSC = 0
    for iter in range(opt.start_epoch, opt.numofIters+1):
        netD.zero_grad()
        netG.zero_grad()
        lossG_G = Variable(torch.FloatTensor([0]).cuda())
        loss_dice = Variable(torch.FloatTensor([0]).cuda())
        lossG_D = Variable(torch.FloatTensor([0]).cuda())
        lossG_D_ADV = Variable(torch.FloatTensor([0]).cuda())
        lossG_focal = Variable(torch.FloatTensor([0]).cuda()) 
        lossG_contour = Variable(torch.FloatTensor([0]).cuda()) 
        lossG_Reg = Variable(torch.FloatTensor([0]).cuda())
        lossD = Variable(torch.FloatTensor([0]).cuda())
        for m in range(0, opt.iter_size):
            if random.random() <0.5:
                loader = data_generator
                labeled = True
            else:
                loader = data_generator_unlabeled
                labeled = False
            
            if labeled: #for labeled cases
                ## step.5.1 prepare training data (input, labels)
                #we should consider different data to train
                if opt.isSegReg:
                    inputs,labels, regGT1, regGT2, regGT3 = data_generator.next()
                elif opt.isContourLoss:
                    inputs,labels,contours = data_generator.next()
                else:
                    inputs,labels = data_generator.next()
                    #print inputs.size,labels.size
                    
                labels = np.squeeze(labels,axis=1)
                labels = labels.astype(int)
                
                if opt.isContourLoss:
                    contours = np.squeeze(contours,axis=1)
                    contours = contours.astype(int)
                    contours = torch.from_numpy(contours)
                    contours = contours.cuda()
                    contours = Variable(contours)
                
                inputs = torch.from_numpy(inputs)
                labels = torch.from_numpy(labels)
                inputs = inputs.cuda()
                labels = labels.cuda()            
                #wrap them into Variable
                inputs, labels = Variable(inputs),Variable(labels)
                
                
                ## step.5.2 prepare the training data for adversarial training
                if opt.isAdLoss:
                    if opt.isSegReg:
                        outputG, outputReg1, outputReg2, outputReg3 = netG(inputs)
                    elif opt.isContourLoss:    
                        outputG,_ = netG(inputs)
                    elif opt.isDeeplySupervised:    
                        outputG,outputG_path1,outputG_path2 = netG(inputs)
                    else:
                        outputG = netG(inputs)
                    #get the probability map of outputG    
#                     if opt.NDim==2:
#                         outputG = softmax2d(outputG) #batch
#                     elif opt.NDim==3:
                    outputG = F.softmax(outputG, dim=1)
            
                    #outputG = outputG.data.max(1)[1]
                    #outputG = torch.squeeze(outputG) #[N,C,W,H]
                    #labels = labels.unsqueeze(1) #expand the 1st dim
                    #one hot encode for the gt labels
                    ohlabels = OneHotEncode(nclass=2,nd=3)(labels.data.cpu()) #NxWxHxD->NxCxWxHxD
                    #print 'outputG: ',outputG.size(),'labels: ',ohlabels.size()
                    outputR = Variable(ohlabels.type(torch.FloatTensor).cuda()) #output_Real
                    outputG = outputG.type(torch.FloatTensor).cuda()
                    
                    if opt.isNetDInputIncludeSource:
                        inputReal_netD = torch.cat((inputs,outputR),dim=1)
                        outputD_real = netD(inputReal_netD)
                        inputFake_netD = torch.cat((inputs,outputG),dim=1)
                        outputD_fake = netD(inputFake_netD)
                    else:
                        outputD_real = netD(outputR)
                        outputD_fake = netD(outputG)
                        
                    
                    ## step.5.2 update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    batch_size = inputs.size(0)
                    output_shape = outputD_real.shape
            
                    if opt.isNetDFullyConv: #if netD is FCN based
                        #train with real data
                        if opt.NDim==3:
                            real_label = torch.ones(output_shape[0],output_shape[2],output_shape[3],output_shape[4]).long()
                            fake_label = torch.zeros(output_shape[0],output_shape[2],output_shape[3],output_shape[4]).long()
                        else:
                            real_label = torch.ones(output_shape[0],output_shape[2],output_shape[3]).long()
                            fake_label = torch.zeros(output_shape[0],output_shape[2],output_shape[3]).long()
                        real_label = Variable(real_label.cuda())
                        loss_real = criterion_BCEND(outputD_real,real_label)
                        loss_real.backward()
                        #train with fake data
                        fake_label = Variable(fake_label.cuda())  
                        loss_fake = criterion_BCEND(outputD_fake,fake_label)
                        loss_fake.backward()      
                    else: #if netD is CNN based
                        #train with real data
                        real_label = torch.ones(batch_size,1)
                        real_label = Variable(real_label.cuda())
                        loss_real = criterion(outputD_real,real_label)
                        loss_real.backward()
                        #train with fake data
                        fake_label=torch.zeros(batch_size,1)
                        fake_label = Variable(fake_label.cuda()) 
                        loss_fake = criterion(outputD_fake,fake_label)
                        loss_fake.backward()
                    
                    lossD = lossD + loss_real + loss_fake
                
                ##step.5.3 update G network: minimize the L1/L2 loss, maximize the D(G(x))
                #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
                #view of equation (note the max and min difference for generator and discriminator)
                if opt.isAdLoss:
                    if opt.isSegReg:
                        outputG, outputReg1, outputReg2, outputReg3 = netG(inputs)
                    elif opt.isContourLoss:
                        outputG,_ = netG(inputs)
                    elif opt.isDeeplySupervised:    
                        outputG,outputG_path1,outputG_path2 = netG(inputs)                    
                    else:
                        outputG = netG(inputs)
                    #obtain the prediction probability maps    
#                     if opt.NDim==2:
#                         outputG = softmax2d(outputG) #batach
#                     elif opt.NDim==3:
                    outputG = F.softmax(outputG, dim=1)
            
        #             outputG = outputG.data.max(1)[1]
                    outputG = outputG.type(torch.FloatTensor).cuda() #we directly use the probabilities: NxCxWxHxD
                    
                    if opt.isNetDInputIncludeSource:
                        input_netD = torch.cat((inputs,outputG),dim=1)
                        outputD = netD(input_netD)
                    else:
                        outputD = netD(outputG)
        
                    #obtain the prediction probability maps    
                    outputD = F.softmax(outputD, dim=1)
                        
                    #compute the non-zero item probability, we have excluded Variable, so it will not involve the training for this part 
                    averProbTensor = (1 - outputD.data[0].cpu()) 
                    
                    if opt.isNetDFullyConv and opt.isConfidenceRegionAware:
                        confRegion = computeVoxelAttentionWeight(averProbTensor)
                        confRegion = Variable(confRegion.cuda(),requires_grad=False)
                    if opt.isSampleImportanceFromAd:
                        averProb = averProbTensor.mean()
            #             print 'prob: ',averProb
                        adImportance = computeSampleAttentionWeight(averProb)
                        
                    if opt.isNetDFullyConv:       
                        lossG_D = lossG_D + opt.lambdaAD * criterion_BCEND(outputD, real_label) #note, for generator, the label for outputG is real
                    else:
                        lossG_D = lossG_D + opt.lambdaAD * criterion(outputD, real_label) #note, for generator, the label for outputG is real
                    lossG_D.backward(retain_graph=True)
                
                if opt.isSegReg:
                    outputG, outputReg1, outputReg2, outputReg3 = netG(inputs)
                elif opt.isContourLoss: 
                    outputG,outputContour = netG(inputs)
                elif opt.isDeeplySupervised:    
                    outputG,outputG_path1,outputG_path2 = netG(inputs)
                else:
                    outputG = netG(inputs) #here I am not sure whether we should use twice or not
                
                if opt.isSoftmaxLoss:
                    if opt.isNetDFullyConv and opt.isConfidenceRegionAware:
                        lossG_G_main = criterion_WCEND(outputG,torch.squeeze(labels,dim=1),confRegion) 
                    elif opt.isSampleImportanceFromAd:
                        lossG_G_main = (1+adImportance) * criterion_CEND(outputG,torch.squeeze(labels,dim=1)) 
                    else:
                        lossG_G_main = criterion_CEND(outputG,torch.squeeze(labels,dim=1)) 
                    if opt.isDeeplySupervised:
                        lossG_G_path1 = opt.lambda_ds1*criterion_CEND(outputG_path1,torch.squeeze(labels,dim=1)) 
                        lossG_G_path2 = opt.lambda_ds2*criterion_CEND(outputG_path2,torch.squeeze(labels,dim=1)) 
                        lossG_G = lossG_G + (lossG_G_main + lossG_G_path1 + lossG_G_path2)
                    else:
                        lossG_G = lossG_G + lossG_G_main
                    lossG_G.backward(retain_graph=True) #compute gradients
            
                if opt.isDiceLoss:
    #             criterion_dice = myWeightedDiceLoss4Organs(organIDs=[0,1,2,3], organWeights=[1,4,8,6])
                    if opt.isSampleImportanceFromAd:
                        loss_dice_main = (1+adImportance) * criterion_dice(outputG,torch.squeeze(labels,dim=1))
                    else:
                        loss_dice_main = criterion_dice(outputG,torch.squeeze(labels,dim=1))
                    if opt.isDeeplySupervised:
                        loss_dice_path1 = opt.lambda_ds1*criterion_dice(outputG_path1,torch.squeeze(labels,dim=1)) 
                        loss_dice_path2 = opt.lambda_ds2*criterion_dice(outputG_path2,torch.squeeze(labels,dim=1)) 
                        loss_dice = loss_dice + (loss_dice_main + loss_dice_path1 + loss_dice_path2)
                    else:
                        loss_dice = loss_dice + loss_dice_main
            #             loss_dice = myDiceLoss4Organs(outputG,torch.squeeze(labels)) #succeed
            #             loss_dice.backward(retain_graph=True) #compute gradients for dice loss
                        loss_dice.backward(retain_graph=True) #compute gradients for dice loss
                
                if opt.isContourLoss:
                    lossG_contour = lossG_contour + criterion_BCE2D(outputContour,contours)
                    lossG_contour.backward(retain_graph=True)
        
                if opt.isSegReg:
                    lossG_Reg1 = criterion_MSE(outputReg1, regGT1)
                    lossG_Reg2 = criterion_MSE(outputReg2, regGT2)
                    lossG_Reg3 = criterion_MSE(outputReg3, regGT3)
                    lossG_Reg = lossG_Reg + lossG_Reg1 + lossG_Reg2 + lossG_Reg3
                    lossG_Reg.backward()
        
                if opt.isFocalLoss:
                    lossG_focal = lossG_focal + criterion_focal(outputG,torch.squeeze(labels,dim=1))
                    lossG_focal.backward(retain_graph=True) #compute gradients        


                #del outputD
                #del outputG
                del inputs
                del labels
            else: #for unlabeled cases
                    #########################################################
                    # Use unlabelled data to get L_semi loss and L_ADV loss #
                    #########################################################
                ## step.5.1 prepare training data (input, labels)
                #we should consider different data to train
                
                inputs_unlabeled = data_generator_unlabeled.next()
            
                inputs_unlabeled = torch.from_numpy(inputs_unlabeled)
                inputs_unlabeled = inputs_unlabeled.cuda()
                
                #wrap them into Variable
                inputs_unlabeled = Variable(inputs_unlabeled)
                            
                ##step.5.3 update G network: minimize the L1/L2 loss, maximize the D(G(x))
                #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
                #view of equation (note the max and min difference for generator and discriminator)
                if opt.isAdLoss:
                    if opt.isSegReg:
                        outputG, outputReg1, outputReg2, outputReg3 = netG(inputs_unlabeled)
                    elif opt.isContourLoss:
                        outputG,_ = netG(inputs_unlabeled)
                    elif opt.isDeeplySupervised:    
                        outputG,outputG_path1,outputG_path2 = netG(inputs_unlabeled)
                    else:
                        outputG = netG(inputs_unlabeled)
                    outputG_nosmax = outputG
                    #obtain the prediction probability maps    
                    outputG = F.softmax(outputG, dim=1)
            
        #             outputG = outputG.data.max(1)[1]
                    outputG = outputG.type(torch.FloatTensor).cuda() #we directly use the probabilities: NxCxWxHxD
                    
                    if opt.isNetDInputIncludeSource:
                        input_netD = torch.cat((inputs_unlabeled,outputG),dim=1)
                        outputD = netD(input_netD)
                    else:
                        outputD = netD(outputG)
                    #obtain the prediction probability maps  
                    outputD = F.softmax(outputD, dim=1)
                    averProbTensor = (1 - outputD.data[0].cpu()) #compute the non-zero item probability
                    averProb = averProbTensor.mean()
        #             print 'prob: ',averProb
                    adImportance = computeSampleAttentionWeight(averProb)
                    
                    ##lets begin to compute  #Adversarial Loss# ##
                    outputD_shape = outputD.size()
                    if opt.NDim==3:
                        real_label = torch.ones(outputD_shape[0],outputD_shape[2],outputD_shape[3],outputD_shape[4]).long()
                    else:
                        real_label = torch.ones(outputD_shape[0],outputD_shape[2],outputD_shape[3]).long()
                    real_label = Variable(real_label.cuda())
                    
                    if opt.isNetDFullyConv:       
                        lossG_D_ADV = lossG_D_ADV + opt.lambdaAD * criterion_BCEND(outputD, real_label) #note, for generator, the label for outputG is real
                    else:
                        lossG_D_ADV = lossG_D_ADV + opt.lambdaAD * criterion(outputD, real_label) #note, for generator, the label for outputG is real
    #                 lossG_D.backward(retain_graph=True)
                    
                    ##lets begin to compute #Semi-Supervised Loss# ##
                    outputG_hardpred = torch.max(outputG,1)[1].squeeze(1) #[0]->max value, [1]->index at the max value which works as Y_hat
    
                    idx = np.zeros(outputG_nosmax.data.cpu().numpy().shape,dtype=np.uint8)
                    if opt.NDim==2:
                        idx = idx.transpose(0, 2, 3, 1) #make channel to the last dimension
                    elif opt.NDim==3:
                        idx = idx.transpose(0, 2, 3, 4, 1) #make channel to the last dimension
                    outputD_np = outputD[:,1,...].data.cpu().numpy()#obtain the probability map to be real
                    outputG_hardpred_np = outputG_hardpred.data.cpu().numpy()
                    opt.th_semi = 1 - currDSC
                    idx[outputD_np > opt.th_semi] = np.identity(opt.out_channels, dtype=idx.dtype)[outputG_hardpred_np[outputD_np > opt.th_semi]]
    #                 idx[outputD_np > opt.th_semi] = outputG_hardpred_np[outputD_np > opt.th_semi]
                    #actually, I now understand what does np.identity work here: it makes NxWxHxDx1->NxWxHxDxC
                    lossG_D = lossG_D_ADV
                    if np.count_nonzero(idx) != 0:
                        outputG_lsmax = nn.LogSoftmax(dim=1)(outputG_nosmax)
                        if opt.NDim==2:
                            idx = Variable(torch.from_numpy(idx.transpose(0,3,1,2)).byte().cuda()) #recover the dimension order to be: NxCxWxH
                        elif opt.NDim==3:
                            idx = Variable(torch.from_numpy(idx.transpose(0,4,1,2,3)).byte().cuda()) #recover the dimension order to be: NxCxWxHxD
                        LGsemi_arr = outputG_lsmax.masked_select(idx)
                        LGsemi = -1*LGsemi_arr.mean()
                        LGsemi = opt.lambdaSEMI*LGsemi
                        lossG_D = lossG_D + LGsemi
    #                     LGsemi.backward()
                    lossG_D.backward(retain_graph=True)
                #del outputD
                #del outputG
                del inputs_unlabeled
                    
        #update network parameters after iter_size weight computation
        optimizerD.step()        
        optimizerG.step() 
        torch.cuda.empty_cache()
        
#         print 'type of lossD: ',type(lossD),' type of lossD/opt.iter_size: ', type(lossD/opt.iter_size)
        if opt.isAdLoss:
            valDSC = showTestStatistics(netG, lossG_G/opt.iter_size, loss_dice/opt.iter_size, lossG_D/opt.iter_size, lossG_focal/opt.iter_size, lossG_contour/opt.iter_size, lossG_Reg/opt.iter_size, lossD/opt.iter_size, iter,labeled=labeled, netD=netD)
        else:
            valDSC = showTestStatistics(netG, lossG_G/opt.iter_size, loss_dice/opt.iter_size, lossG_D/opt.iter_size, lossG_focal/opt.iter_size, lossG_contour/opt.iter_size, lossG_Reg/opt.iter_size, lossD/opt.iter_size, iter,labeled=labeled)
        #showTestStatistics(netG, lossG_G, loss_dice, lossG_D, lossG_focal, lossG_contour, lossG_Reg, lossD, iter,labeled=labeled)
        if valDSC!=-1:
            currDSC = valDSC
            
        if iter%opt.decLREvery==0 and iter>0:
            opt.lr = opt.lr*0.1
            adjust_learning_rate(optimizerG, opt.lr)
            print 'now the learning rate is {}'.format(opt.lr)
        
    print('Finished Training')
    
if __name__ == '__main__':
#     testGradients()     
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuID 
    main()
    
