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

import ast

def arg_as_list(s):                                                            
    v = ast.literal_eval(s)                                                    
    if type(v) is not list:                                                    
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v  


# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--isSemiSupervised", action="store_true", help="is the training semi-supervised?", default=False)
parser.add_argument("--NDim", type=int, default=3, help="the dimension of the shape, 1D, 2D or 3D?")
parser.add_argument("--in_channels", type=int, default=1, help="the input channels ?")
parser.add_argument("--out_channels", type=int, default=2, help="the output channels (num of classes)?")
parser.add_argument("--isNetDFullyConv", action="store_true", help="is the netD fully convolutional network?", default=True)
parser.add_argument("--isNetDInputIncludeSource", action="store_true", help="is the input of netD including the source image or not?", default=False)
parser.add_argument("--in_channels_netD", type=int, default=2, help="the input channels for netD?")
parser.add_argument("--out_channels_netD", type=int, default=2, help="the output channels for netD?")
parser.add_argument("--given_weight", type=arg_as_list, default=[1,8], help="the given weight for each organ (bg included)")
parser.add_argument("--given_ids", type=arg_as_list, default=[0,1], help="the given organ id for each organ (bg included)")
parser.add_argument("--input_sz", type=arg_as_list, default=[16,64,64], help="the input patch size of list")
parser.add_argument("--output_sz", type=arg_as_list, default=[16,64,64], help="the output patch size of list")
parser.add_argument("--test_step_sz", type=arg_as_list, default=[1,8,8], help="the step size at testing one subject")
parser.add_argument("--isSegReg", action="store_true", help="is Seg and Reg?", default=False)
parser.add_argument("--isDiceLoss", action="store_true", help="is Dice Loss used?", default=True)
parser.add_argument("--isSoftmaxLoss", action="store_true", help="is Softmax Loss used?", default=True)
parser.add_argument("--isContourLoss", action="store_true", help="is Contour Loss used?", default=False)
parser.add_argument("--isResidualEnhancement", action="store_true", help="is residual learning operation enhanced?", default=True)
parser.add_argument("--isViewExpansion", action="store_true", help="is view expanded?", default=True)
parser.add_argument("--isAdLoss", action="store_true", help="is adversarial loss used?", default=True)
parser.add_argument("--isSpatialDropOut", action="store_true", help="is spatial dropout used?", default=False)
parser.add_argument("--isFocalLoss", action="store_true", help="is focal loss used?", default=False)
parser.add_argument("--isSampleImportanceFromAd", action="store_true", help="is sample importance from adversarial network used?", default=False)
parser.add_argument("--dropoutRate", type=float, default=0.25, help="Spatial Dropout Rate. Default=0.25")
parser.add_argument("--lambdaAD", type=float, default=0.1, help="loss coefficient for AD loss. Default=0")
parser.add_argument("--adImportance", type=float, default=0, help="Sample importance from AD network. Default=0")

parser.add_argument("--how2normalize", type=int, default=4, help="how to normalize the data")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--numofIters", type=int, default=200000, help="number of iterations to train for")
parser.add_argument("--showTrainLossEvery", type=int, default=100, help="number of iterations to show train loss")
parser.add_argument("--saveModelEvery", type=int, default=5000, help="number of iterations to save the model")
parser.add_argument("--showTestPerformanceEvery", type=int, default=1000, help="number of iterations to show test performance")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--lr_netD", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--decLREvery", type=int, default=100000, help="Sets the learning rate to the initial LR decayed by momentum every n iterations, Default: n=40000")
parser.add_argument("--cuda", action="store_true", help="Use cuda?", default=True)
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--prefixModelName", default="/home/dongnie/Desktop/myPyTorch/pytorch-SRResNet23D/SegCha_3D_wce_wdice_viewExp_resEnhance_lrdcr_fullAd_0112_", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="preSub45_cha_3D_wce_wdice_viewExp_resEhance_lrdcr_fullAd_0112_", type=str, help="prefix of the to-be-saved predicted filename")

# parser.add_argument("--path_test",default='/shenlab/lab_stor5/dongnie/challengeData/data',type=str, help="the path for the testing nii.gz files")
# parser.add_argument("--path_patients_h5",default='/shenlab/lab_stor5/dongnie/challengeData/pelvicSegRegContourBatchH5',type=str, help="the path for the training hdf5 files")
# parser.add_argument("--path_patients_h5_test",default='/shenlab/lab_stor5/dongnie/challengeData/pelvicSegRegContourH5Test',type=str, help="the path for the testing hdf5 files")

parser.add_argument("--path_test",default='/home/dongnie/warehouse/pelvicSeg/prostateChallenge/data',type=str, help="the path for the testing nii.gz files")
parser.add_argument("--path_patients_h5",default='/home/dongnie/warehouse/pelvicSeg/prostateChallenge/pelvic3DSegRegContourBatchH5',type=str, help="the path for the training hdf5 files")
parser.add_argument("--path_patients_h5_test",default='/home/dongnie/warehouse/pelvicSeg/prostateChallenge/pelvic3DSegRegContourH5Test',type=str, help="the path for the testing hdf5 files")


def main():    

########################################configs####################################
    global opt, model, running_loss, start, criterion_dice, criterion_CEND, data_generator_test, path_test
    opt = parser.parse_args()
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



    if opt.isSegReg:
        negG = ResSegRegNet(opt.in_channels, opt.out_channels, nd=opt.NDim)
    elif opt.isContourLoss:
        netG = ResSegContourNet(opt.in_channels, opt.out_channels, nd=opt.NDim, isRandomConnection=opt.isResidualEnhancement,isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut,dropoutRate=opt.dropoutRate)
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

    
    
    criterion_MSE = nn.MSELoss()
    
#     criterion_NLL2D = nn.NLLLoss2d(weight=given_weight)
    if opt.NDim==2:
        criterion_CEND = CrossEntropy2d(weight=given_weight)
    elif opt.NDim==3: 
        criterion_CEND = CrossEntropy3d(weight=given_weight)
        criterion_BCEND = CrossEntropy3d()

    
    criterion_BCE2D = CrossEntropy2d()#for contours
    
#     criterion_dice = DiceLoss4Organs(organIDs=[1,2,3], organWeights=[1,1,1])
#     criterion_dice = WeightedDiceLoss4Organs()
    criterion_dice = myWeightedDiceLoss4Organs(organIDs=given_ids, organWeights = given_weight)
    
    criterion_focal = myFocalLoss(4, alpha=given_weight, gamma=2)
    
    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    criterion_dice = criterion_dice.cuda()
    criterion_MSE = criterion_MSE.cuda()
    criterion_CEND = criterion_CEND.cuda()
    criterion_BCEND = criterion_BCEND.cuda() #for discriminator
    criterion_BCE2D = criterion_BCE2D.cuda()
    criterion_focal = criterion_focal.cuda()
    softmax2d = nn.Softmax2d()
#     inputs=Variable(torch.randn(1000,1,32,32)) #here should be tensor instead of variable
#     targets=Variable(torch.randn(1000,10,1,1)) #here should be tensor instead of variable
#     trainset=data_utils.TensorDataset(inputs, targets)
#     trainloader = data_utils.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#     inputs=torch.randn(1000,1,32,32)
#     targets=torch.LongTensor(1000)
    


#     batch_size = 10
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
    
    
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            netG.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
    running_loss = 0.0
    start = time.time()
    for iter in range(opt.start_epoch, opt.numofIters+1):
        #print('iter %d'%iter)
        
        if opt.isSegReg:
            inputs,labels, regGT1, regGT2, regGT3 = data_generator.next()
        elif opt.isContourLoss:
            inputs,labels,contours = data_generator.next()
        else:
            inputs,labels = data_generator.next()
            #print inputs.size,labels.size

        labels = np.squeeze(labels)
        labels = labels.astype(int)
        
        if opt.isContourLoss:
            contours = np.squeeze(contours)
            contours = contours.astype(int)
            contours = torch.from_numpy(contours)
            contours = contours.cuda()
            contours = Variable(contours)
        
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        inputs = inputs.cuda()
        labels = labels.cuda()
        #we should consider different data to train
        
        #wrap them into Variable
        inputs,labels = Variable(inputs),Variable(labels)
        
        if opt.isAdLoss:
            #zero the parameter gradients
            #netD.zero_grad()
            #forward + backward +optimizer
            if opt.isSegReg:
                outputG, outputReg1, outputReg2, outputReg3 = netG(inputs)
            elif opt.isContourLoss:    
                outputG,_ = netG(inputs)
            else:
                outputG = netG(inputs)
                
                
            if opt.NDim==2:
                outputG = softmax2d(outputG) #batch
            elif opt.NDim==3:
                outputG = F.softmax(outputG, dim=1)
    
    
            #outputG = outputG.data.max(1)[1]
            #outputG = torch.squeeze(outputG) #[N,C,W,H]
            #labels = labels.unsqueeze(1) #expand the 1st dim
#             ohlabels = OneHotEncode(nclass=2,nd=3)(labels.data.cpu()) #NxWxHxD->NxCxWxHxD
            ohlabels = OneHotEncode(nclass=opt.out_channels,nd=opt.NDim)(labels.data.cpu()) #NxWxHxD->NxCxWxHxD

            #print 'outputG: ',outputG.size(),'labels: ',ohlabels.size()
            outputR = Variable(ohlabels.type(torch.FloatTensor).cuda()) #output_Real
            outputG = outputG.type(torch.FloatTensor).cuda()
            
            if opt.isNetDInputIncludeSource:
                inputReal_netD = torch.cat((inputs,outputR),dim=1)
                outputD_real = netD(inputReal_netD)
                inputFake_netD = torch.cat((inputs,outputG.unsqueeze(1)),dim=1)
                outputD_fake = netD(inputFake_netD)
            else:
                outputD_real = netD(outputR)
                outputD_fake = netD(outputG)
                
    
            
            ## (1) update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            batch_size = inputs.size(0)
            output_shape = outputD_real.shape
            #print(inputs.size())
    
            if opt.isNetDFullyConv:
                #train with real data
                if opt.NDim==3:
                    real_label = torch.ones(output_shape[0],output_shape[2],output_shape[3],output_shape[4]).long()
                    fake_label = torch.zeros(output_shape[0],output_shape[2],output_shape[3],output_shape[4]).long()
                else:
                    real_label = torch.ones(output_shape[0],output_shape[2],output_shape[3]).long()
                    fake_label = torch.zeros(output_shape[0],output_shape[2],output_shape[3]).long()
                real_label = real_label.cuda()
                #print(real_label.size())
                real_label = Variable(real_label)
                #print(outputD_real.size())
                loss_real = criterion_BCEND(outputD_real,real_label)
                loss_real.backward()
                #train with fake data
                
                fake_label = fake_label.cuda()
                fake_label = Variable(fake_label)
                
                loss_fake = criterion_BCEND(outputD_fake,fake_label)
                loss_fake.backward()
            else:
                #train with real data
                real_label = torch.ones(batch_size,1)
                real_label = real_label.cuda()
                #print(real_label.size())
                real_label = Variable(real_label)
                #print(outputD_real.size())
                loss_real = criterion(outputD_real,real_label)
                loss_real.backward()
                #train with fake data
                fake_label=torch.zeros(batch_size,1)
                fake_label = fake_label.cuda()
                fake_label = Variable(fake_label)
                
                loss_fake = criterion(outputD_fake,fake_label)
                loss_fake.backward()
            
            lossD = loss_real + loss_fake
    
            optimizerD.step()
        
        ## (2) update G network: minimize the L1/L2 loss, maximize the D(G(x))
        
                #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
        #angel of equation (note the max and min difference for generator and discriminator)
        if opt.isAdLoss:
            if opt.isSegReg:
                outputG, outputReg1, outputReg2, outputReg3 = netG(inputs)
            elif opt.isContourLoss:
                outputG,_ = netG(inputs)
            else:
                outputG = netG(inputs)
                
            if opt.NDim==2:
                outputG = softmax2d(outputG) #batach
            elif opt.NDim==3:
                outputG = F.softmax(outputG, dim=1)
            
                
#             outputG = outputG.data.max(1)[1]
            outputG = outputG.type(torch.FloatTensor).cuda() #we directly use the probabilities: NxCxWxHxD
#         print 'outputG shape, ',outputG.size()
            
            if opt.isNetDInputIncludeSource:
                input_netD = torch.cat((inputs,outputG.unsqueeze(1)),dim=1)
                outputD = netD(input_netD)
            else:
                outputD = netD(outputG)

            averProb = outputD.data.cpu().mean()
#             print 'prob: ',averProb
#             adImportance = computeAttentionWeight(averProb)
            adImportance = computeSampleAttentionWeight(averProb)       
            lossG_D = opt.lambdaAD * criterion_BCEND(outputD, real_label) #note, for generator, the label for outputG is real
            lossG_D.backward(retain_graph=True)
        
        if opt.isSegReg:
            outputG, outputReg1, outputReg2, outputReg3 = netG(inputs)
        elif opt.isContourLoss: 
            outputG,outputContour = netG(inputs)
        else:
            outputG = netG(inputs) #here I am not sure whether we should use twice or not
        netG.zero_grad()
        
        if opt.isFocalLoss:
            lossG_focal = criterion_focal(outputG,torch.squeeze(labels))
            lossG_focal.backward(retain_graph=True) #compute gradients
        
        if opt.isSoftmaxLoss:
            if opt.isSampleImportanceFromAd:
                lossG_G = (1+adImportance) * criterion_CEND(outputG,torch.squeeze(labels)) 
            else:
                lossG_G = criterion_CEND(outputG,torch.squeeze(labels)) 
                
            lossG_G.backward(retain_graph=True) #compute gradients
        
        if opt.isContourLoss:
            lossG_contour = criterion_BCE2D(outputContour,contours)
            lossG_contour.backward(retain_graph=True)
#         criterion_dice(outputG,torch.squeeze(labels))
#         print 'hahaN'
        if opt.isSegReg:
            lossG_Reg1 = criterion_MSE(outputReg1, regGT1)
            lossG_Reg2 = criterion_MSE(outputReg2, regGT2)
            lossG_Reg3 = criterion_MSE(outputReg3, regGT3)
            lossG_Reg = lossG_Reg1 + lossG_Reg2 + lossG_Reg3
            lossG_Reg.backward()

        if opt.isDiceLoss:
#             print 'isDiceLoss line278'
#             criterion_dice = myWeightedDiceLoss4Organs(organIDs=[0,1,2,3], organWeights=[1,4,8,6])
            if opt.isSampleImportanceFromAd:
                loss_dice = (1+adImportance) * criterion_dice(outputG,torch.squeeze(labels))
            else:
                loss_dice = criterion_dice(outputG,torch.squeeze(labels))
#             loss_dice = myDiceLoss4Organs(outputG,torch.squeeze(labels)) #succeed
#             loss_dice.backward(retain_graph=True) #compute gradients for dice loss
            loss_dice.backward() #compute gradients for dice loss
        
        #lossG_D.backward()

        
        #for other losses, we can define the loss function following the pytorch tutorial
        
        optimizerG.step() #update network parameters
#         print 'gradients of parameters****************************'
#         [x.grad.data for x in netG.parameters()]
#         print x.grad.data[0]
#         print '****************************'
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
            lossG = loss_dice.data[0]

        #print('loss for generator is %f'%lossG.data[0])
        #print statistics
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
            torch.save(netG.state_dict(), opt.prefixModelName+'%d.pt'%iter)
            print 'save model: '+opt.prefixModelName+'%d.pt'%iter
        
        if iter%opt.decLREvery==0 and iter>0:
            opt.lr = opt.lr*0.1
            adjust_learning_rate(optimizerG, opt.lr)
            print 'now the learning rate is {}'.format(opt.lr)
        
        if iter%opt.showTestPerformanceEvery==0: #test one subject  
            # to test on the validation dataset in the format of h5 
            inputs,labels = data_generator_test.next()
            labels = np.squeeze(labels)
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
            else:
                outputG = netG(inputs) #here I am not sure whether we should use twice or not
            
            lossG_G = criterion_CEND(outputG,torch.squeeze(labels))
            loss_dice = criterion_dice(outputG,torch.squeeze(labels))
            del outputG
            print '.......come to validation stage: iter {}'.format(iter),'........'
            print 'lossG_G and loss_dice are %.2f and %.2f respectively.'%(lossG_G.data[0], loss_dice.data[0])
            
            ####release all the unoccupied memory####
            torch.cuda.empty_cache()

            mr_test_itk=sitk.ReadImage(os.path.join(path_test,'Case45.nii.gz'))
            ct_test_itk=sitk.ReadImage(os.path.join(path_test,'Case45_segmentation.nii.gz'))
            
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
#             x1=80
#             x2=192
#             y1=35
#             y2=235
#             matFA = mrnp[:,y1:y2,x1:x2] #note, matFA and matFAOut same size 
#             matGT = ctnp[:,y1:y2,x1:x2]
            matFA = mrnp
            matGT = ctnp
#                 volFA = sitk.GetImageFromArray(matFA)
#                 sitk.WriteImage(volFA,'volFA'+'.nii.gz')
#                 volGT = sitk.GetImageFromArray(matGT)
#                 sitk.WriteImage(volGT,'volGT'+'.nii.gz')
            
            matOut,_ = testOneSubject(matFA,matGT, opt.out_channels, opt.input_sz, opt.output_sz, opt.test_step_sz, netG,opt.prefixModelName+'%d.pt'%iter, nd=opt.NDim)
            ct_estimated = np.zeros([ctnp.shape[0],ctnp.shape[1],ctnp.shape[2]])
            print 'matOut shape: ',matOut.shape
#             ct_estimated[:,y1:y2,x1:x2] = matOut
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
        
    print('Finished Training')
    

'''
    A function to show the test statistics during training
'''
def showTestStatistics(netG, lossG_G, loss_dice, lossG_D, lossG_focal, lossG_contour, lossG_Reg, lossD, iter):
    
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
        lossG = loss_dice.data[0]

    #print('loss for generator is %f'%lossG.data[0])
    #print statistics
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
        torch.save(netG.state_dict(), opt.prefixModelName+'%d.pt'%iter)
        print 'save model: '+opt.prefixModelName+'%d.pt'%iter
    
    if iter%opt.decLREvery==0 and iter>0:
        opt.lr = opt.lr*0.1
        adjust_learning_rate(optimizerG, opt.lr)
        print 'now the learning rate is {}'.format(opt.lr)
    
    if iter%opt.showTestPerformanceEvery==0: #test one subject  
        # to test on the validation dataset in the format of h5 
        inputs,labels = data_generator_test.next()
        labels = np.squeeze(labels)
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
        else:
            outputG = netG(inputs) #here I am not sure whether we should use twice or not
        
        lossG_G = criterion_CEND(outputG,torch.squeeze(labels))
        loss_dice = criterion_dice(outputG,torch.squeeze(labels))
        del outputG
        print '.......come to validation stage: iter {}'.format(iter),'........'
        print 'lossG_G and loss_dice are %.2f and %.2f respectively.'%(lossG_G.data[0], loss_dice.data[0])
        
        ####release all the unoccupied memory####
        torch.cuda.empty_cache()

        mr_test_itk=sitk.ReadImage(os.path.join(path_test,'Case45.nii.gz'))
        ct_test_itk=sitk.ReadImage(os.path.join(path_test,'Case45_segmentation.nii.gz'))
        
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
#             x1=80
#             x2=192
#             y1=35
#             y2=235
#             matFA = mrnp[:,y1:y2,x1:x2] #note, matFA and matFAOut same size 
#             matGT = ctnp[:,y1:y2,x1:x2]
        matFA = mrnp
        matGT = ctnp
#                 volFA = sitk.GetImageFromArray(matFA)
#                 sitk.WriteImage(volFA,'volFA'+'.nii.gz')
#                 volGT = sitk.GetImageFromArray(matGT)
#                 sitk.WriteImage(volGT,'volGT'+'.nii.gz')
        
        matOut,_ = testOneSubject(matFA,matGT, opt.out_channels, opt.input_sz, opt.output_sz, opt.test_step_sz, netG,opt.prefixModelName+'%d.pt'%iter, nd=opt.NDim)
        ct_estimated = np.zeros([ctnp.shape[0],ctnp.shape[1],ctnp.shape[2]])
        print 'matOut shape: ',matOut.shape
#             ct_estimated[:,y1:y2,x1:x2] = matOut
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
    path_patients_h5_test = opt.path_patients_h5_test
########################################configs####################################



    if opt.isSegReg:
        negG = ResSegRegNet(opt.in_channels, opt.out_channels, nd=opt.NDim)
    elif opt.isContourLoss:
        netG = ResSegContourNet(opt.in_channels, opt.out_channels, nd=opt.NDim, isRandomConnection=opt.isResidualEnhancement,isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut,dropoutRate=opt.dropoutRate)
    else:
        netG = ResSegNet(opt.in_channels, opt.out_channels, nd=opt.NDim, isRandomConnection=opt.isResidualEnhancement,isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut,dropoutRate=opt.dropoutRate)
    #netG.apply(weights_init)
    netG.cuda()
    
    if opt.isAdLoss:
        netD = Discriminator(1, nd=opt.NDim)
        netD.apply(weights_init)
        netD.cuda()
        optimizerD =optim.Adam(netD.parameters(),lr=opt.lr)
    
    params = list(netG.parameters())
    print('len of params is ')
    print(len(params))
    print('size of params is ')
    print(params[0].size())
    
    optimizerG =optim.Adam(netG.parameters(),lr=opt.lr)

    
    
    criterion_MSE = nn.MSELoss()
    
#     criterion_NLL2D = nn.NLLLoss2d(weight=given_weight)
    if opt.NDim==2:
        criterion_CEND = CrossEntropy2d(weight=given_weight)
    elif opt.NDim==3: 
        criterion_CEND = CrossEntropy3d(weight=given_weight)
        criterion_BCEND = CrossEntropy3d()
    
    criterion_BCE2D = CrossEntropy2d()#for contours
    
#     criterion_dice = DiceLoss4Organs(organIDs=[1,2,3], organWeights=[1,1,1])
#     criterion_dice = WeightedDiceLoss4Organs()
    criterion_dice = myWeightedDiceLoss4Organs(organIDs=given_ids, organWeights = given_weight)
    
    criterion_focal = myFocalLoss(4, alpha=given_weight, gamma=2)
    
    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    criterion_dice = criterion_dice.cuda()
    criterion_MSE = criterion_MSE.cuda()
    criterion_CEND = criterion_CEND.cuda()
    criterion_BCEND = criterion_BCEND.cuda()
    criterion_BCE2D = criterion_BCE2D.cuda()
    criterion_focal = criterion_focal.cuda()
    softmax2d = nn.Softmax2d()
#     inputs=Variable(torch.randn(1000,1,32,32)) #here should be tensor instead of variable
#     targets=Variable(torch.randn(1000,10,1,1)) #here should be tensor instead of variable
#     trainset=data_utils.TensorDataset(inputs, targets)
#     trainloader = data_utils.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#     inputs=torch.randn(1000,1,32,32)
#     targets=torch.LongTensor(1000)
    


#     batch_size = 10
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
    
    
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            netG.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
    running_loss = 0.0
    start = time.time()
    for iter in range(opt.start_epoch, opt.numofIters+1):
        #print('iter %d'%iter)
        
        if opt.isSegReg:
            inputs,labels, regGT1, regGT2, regGT3 = data_generator.next()
        elif opt.isContourLoss:
            inputs,labels,contours = data_generator.next()
        else:
            inputs,labels = data_generator.next()
            #print inputs.size,labels.size

        labels = np.squeeze(labels)
        labels = labels.astype(int)
        
        if opt.isContourLoss:
            contours = np.squeeze(contours)
            contours = contours.astype(int)
            contours = torch.from_numpy(contours)
            contours = contours.cuda()
            contours = Variable(contours)
        
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        inputs = inputs.cuda()
        labels = labels.cuda()
        #we should consider different data to train
        
        #wrap them into Variable
        inputs,labels = Variable(inputs),Variable(labels)
        
        if opt.isAdLoss:
            #zero the parameter gradients
            #netD.zero_grad()
            #forward + backward +optimizer
            if opt.isSegReg:
                outputG, outputReg1, outputReg2, outputReg3 = netG(inputs)
            elif opt.isContourLoss:    
                outputG,_ = netG(inputs)
            else:
                outputG = netG(inputs)
                
                
            if opt.NDim==2:
                outputG = softmax2d(outputG) #batach
            elif opt.NDim==3:
                outputG = F.softmax(outputG, dim=1)
    
    
#             outputG = outputG.data.max(1)[1]
            #outputG = torch.squeeze(outputG) #[N,C,W,H]
            labels = labels.unsqueeze(1) #expand the 1st dim
    #         print 'outputG: ',outputG.size(),'labels: ',labels.size()
            outputR = labels.type(torch.FloatTensor).cuda() #output_Real
            outputG = Variable(outputG.type(torch.FloatTensor).cuda()) # we directly use the probabilities: NxCxWxHxD
            outputD_real = netD(outputR)
    #         print 'size outputG: ',outputG.unsqueeze(1).size()
            outputD_fake = netD(outputG.unsqueeze(1))
    
            
            ## (1) update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            batch_size = inputs.size(0)
            #print(inputs.size())
            #train with real data
    #         real_label = torch.FloatTensor(batch_size)
    #         real_label.data.resize_(batch_size).fill_(1)
            real_label = torch.ones(batch_size,1)
            real_label = real_label.cuda()
            #print(real_label.size())
            real_label = Variable(real_label)
            #print(outputD_real.size())
            loss_real = criterion(outputD_real,real_label)
            loss_real.backward()
            #train with fake data
            fake_label=torch.zeros(batch_size,1)
    #         fake_label = torch.FloatTensor(batch_size)
    #         fake_label.data.resize_(batch_size).fill_(0)
            fake_label = fake_label.cuda()
            fake_label = Variable(fake_label)
            loss_fake = criterion(outputD_fake,fake_label)
            loss_fake.backward()
            
            lossD = loss_real + loss_fake
    
            optimizerD.step()
        
        ## (2) update G network: minimize the L1/L2 loss, maximize the D(G(x))
        
                #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
        #angel of equation (note the max and min difference for generator and discriminator)
        if opt.isAdLoss:
            if opt.isSegReg:
                outputG, outputReg1, outputReg2, outputReg3 = netG(inputs)
            elif opt.isContourLoss:
                outputG,_ = netG(inputs)
            else:
                outputG = netG(inputs)
            outputG = outputG.data.max(1)[1]
            outputG = Variable(outputG.type(torch.FloatTensor).cuda())
#         print 'outputG shape, ',outputG.size()

            outputD = netD(outputG.unsqueeze(1))
            averProb = outputD.data.cpu().mean()
#             print 'prob: ',averProb
#             adImportance = computeAttentionWeight(averProb)
            adImportance = computeSampleAttentionWeight(averProb)       
            lossG_D = opt.lambdaAD * criterion(outputD, real_label) #note, for generator, the label for outputG is real
            lossG_D.backward(retain_graph=True)
        
        if opt.isSegReg:
            outputG, outputReg1, outputReg2, outputReg3 = netG(inputs)
        elif opt.isContourLoss: 
            outputG,outputContour = netG(inputs)
        else:
            outputG = netG(inputs) #here I am not sure whether we should use twice or not
        netG.zero_grad()
        
        if opt.isFocalLoss:
            lossG_focal = criterion_focal(outputG,torch.squeeze(labels))
            lossG_focal.backward(retain_graph=True) #compute gradients
        
        if opt.isSoftmaxLoss:
            if opt.isSampleImportanceFromAd:
                lossG_G = (1+adImportance) * criterion_CEND(outputG,torch.squeeze(labels)) 
            else:
                lossG_G = criterion_CEND(outputG,torch.squeeze(labels)) 
                
            lossG_G.backward(retain_graph=True) #compute gradients
        
        if opt.isContourLoss:
            lossG_contour = criterion_BCE2D(outputContour,contours)
            lossG_contour.backward(retain_graph=True)
#         criterion_dice(outputG,torch.squeeze(labels))
#         print 'hahaN'
        if opt.isSegReg:
            lossG_Reg1 = criterion_MSE(outputReg1, regGT1)
            lossG_Reg2 = criterion_MSE(outputReg2, regGT2)
            lossG_Reg3 = criterion_MSE(outputReg3, regGT3)
            lossG_Reg = lossG_Reg1 + lossG_Reg2 + lossG_Reg3
            lossG_Reg.backward()

        if opt.isDiceLoss:
#             print 'isDiceLoss line278'
#             criterion_dice = myWeightedDiceLoss4Organs(organIDs=[0,1,2,3], organWeights=[1,4,8,6])
            if opt.isSampleImportanceFromAd:
                loss_dice = (1+adImportance) * criterion_dice(outputG,torch.squeeze(labels))
            else:
                loss_dice = criterion_dice(outputG,torch.squeeze(labels))
#             loss_dice = myDiceLoss4Organs(outputG,torch.squeeze(labels)) #succeed
#             loss_dice.backward(retain_graph=True) #compute gradients for dice loss
            loss_dice.backward() #compute gradients for dice loss
        
        #lossG_D.backward()

        
        #for other losses, we can define the loss function following the pytorch tutorial
        
        optimizerG.step() #update network parameters
#         print 'gradients of parameters****************************'
#         [x.grad.data for x in netG.parameters()]
#         print x.grad.data[0]
#         print '****************************'
        if opt.isDiceLoss and opt.isSoftmaxLoss and opt.isAdLoss and opt.isSegReg and opt.isFocalLoss:
            lossG = opt.lambdaAD * lossG_D + lossG_G+loss_dice.data[0] + lossG_Reg + lossG_focal
        if opt.isDiceLoss and opt.isSoftmaxLoss and opt.isSegReg and opt.isFocalLoss:
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
            lossG = loss_dice.data[0]

        #print('loss for generator is %f'%lossG.data[0])
        #print statistics
        running_loss = running_loss + lossG.data[0]
#         print 'running_loss is ',running_loss,' type: ',type(running_loss)
        
#         print type(outputD_fake.cpu().data[0].numpy())
        
        if iter%opt.showTrainLossEvery==0: #print every 2000 mini-batches
            print '************************************************'
            print 'time now is: ' + time.asctime(time.localtime(time.time()))
            if opt.isAdLoss:
                print 'the outputD_real for iter {}'.format(iter), ' is ',outputD_real.cpu().data[0].numpy()[0]
                print 'the outputD_fake for iter {}'.format(iter), ' is ',outputD_fake.cpu().data[0].numpy()[0]
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
            torch.save(netG.state_dict(), opt.prefixModelName+'%d.pt'%iter)
            print 'save model: '+opt.prefixModelName+'%d.pt'%iter
        
        if iter%opt.decLREvery==0 and iter>0:
            opt.lr = opt.lr*0.1
            adjust_learning_rate(optimizerG, opt.lr)
            print 'now the learning rate is {}'.format(opt.lr)
        
        if iter%opt.showTestPerformanceEvery==0: #test one subject  
            # to test on the validation dataset in the format of h5 
            inputs,labels = data_generator_test.next()
            labels = np.squeeze(labels)
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
            else:
                outputG = netG(inputs) #here I am not sure whether we should use twice or not
            
            lossG_G = criterion_CEND(outputG,torch.squeeze(labels))
            loss_dice = criterion_dice(outputG,torch.squeeze(labels))
            print '.......come to validation stage: iter {}'.format(iter),'........'
            print 'lossG_G and loss_dice are %.2f and %.2f respectively.'%(lossG_G.data[0], loss_dice.data[0])
            
            ####release all the unoccupied memory####
            torch.cuda.empty_cache()
            del outputG

            mr_test_itk=sitk.ReadImage(os.path.join(path_test,'Case45.nii.gz'))
            ct_test_itk=sitk.ReadImage(os.path.join(path_test,'Case45_segmentation.nii.gz'))
            
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
#             x1=80
#             x2=192
#             y1=35
#             y2=235
#             matFA = mrnp[:,y1:y2,x1:x2] #note, matFA and matFAOut same size 
#             matGT = ctnp[:,y1:y2,x1:x2]
            matFA = mrnp
            matGT = ctnp
#                 volFA = sitk.GetImageFromArray(matFA)
#                 sitk.WriteImage(volFA,'volFA'+'.nii.gz')
#                 volGT = sitk.GetImageFromArray(matGT)
#                 sitk.WriteImage(volGT,'volGT'+'.nii.gz')
            
            matOut,_ = testOneSubject(matFA,matGT, opt.out_channels, opt.input_sz, opt.output_sz, opt.test_step_sz, netG,opt.prefixModelName+'%d.pt'%iter, nd=opt.NDim)
            ct_estimated = np.zeros([ctnp.shape[0],ctnp.shape[1],ctnp.shape[2]])
            print 'matOut shape: ',matOut.shape
#             ct_estimated[:,y1:y2,x1:x2] = matOut
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
        
    print('Finished Training')
    
if __name__ == '__main__':
#     testGradients()     
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
    
