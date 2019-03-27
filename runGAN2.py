# from __future__ import print_function
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
import time

def testGradients():
    your_input_size = [10,4,168,112]
    your_target_size = [10,1,168,112]
    input = Variable(torch.rand(your_input_size))
    target = Variable(torch.rand(your_target_size))
    save = True
    res = torch.autograd.gradcheck(WeightedDiceLoss4Organs(), (input, target))
    print(res) # res should be True if the gradients are correct.

def main():    
    #netG=Generator()
#     netG=UNet()
    isSegReg = False
    isSpatialDropout = True
    
    if isSegReg:
        negG = ResSegRegNet()
    else:
        netG = ResSegNet(isSpatialDropout)
    #netG.apply(weights_init)
    netG.cuda()
    
    netD = Discriminator()
    netD.apply(weights_init)
    netD.cuda()
    
    params = list(netG.parameters())
    print('len of params is ')
    print(len(params))
    print('size of params is ')
    print(params[0].size())
    
#     optimizerG =optim.SGD(netG.parameters(),lr=1e-2)
    optimizerG =optim.Adam(netG.parameters(),lr=1e-4)

#     optimizerD =optim.SGD(netD.parameters(),lr=1e-4)
    optimizerD =optim.Adam(netD.parameters(),lr=1e-4)
    
    criterion_MSE = nn.MSELoss()
    given_weight = torch.FloatTensor([1,4,8,6])
    given_weight = given_weight.cuda()
#     criterion_NLL2D = nn.NLLLoss2d(weight=given_weight)
    criterion_CE2D = CrossEntropy2d(weight=given_weight)
#     criterion_dice = DiceLoss4Organs(organIDs=[1,2,3], organWeights=[1,1,1])
#     criterion_dice = WeightedDiceLoss4Organs()
    criterion_dice = myWeightedDiceLoss4Organs(organIDs=[0,1,2,3], organWeights=[1,4,8,6])
    criterion = nn.BCELoss()
    criterion.cuda()
    criterion_dice.cuda()
    criterion_MSE.cuda()
    criterion_CE2D.cuda()
    softmax2d = nn.Softmax2d()
    #inputs=Variable(torch.randn(1000,1,32,32)) #here should be tensor instead of variable
    #targets=Variable(torch.randn(1000,10,1,1)) #here should be tensor instead of variable
#     trainset=data_utils.TensorDataset(inputs, targets)
#     trainloader = data_utils.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#     inputs=torch.randn(1000,1,32,32)
#     targets=torch.LongTensor(1000)
    
    path_test = '/home/dongnie/warehouse/mrs_data'
    path_patients_h5 = '/home/dongnie/warehouse/BrainEstimation/brainH5'
    path_patients_h5 = '/home/dongnie/warehouse/pelvicSeg/pelvicH5'
    path_patients_h5 = '/home/dongnie/warehouse/pelvicSeg/pelvicSegRegH5'
    path_patients_h5 = '/home/dongnie/warehouse/pelvicSeg/pelvicSegRegBatchH5'
    path_patients_h5 = '/home/dongnie/warehouse/pelvicSeg/pelvicSegRegPartH5/' #only contains 1-15
    path_patients_h5_test ='/home/dongnie/warehouse/pelvicSeg/pelvicSegRegH5Test'
    batch_size = 10
    if isSegReg:
        data_generator = Generator_2D_slices_variousKeys(path_patients_h5,batch_size,inputKey='dataMR2D',outputKey='dataSeg2D',regKey1='dataBladder2D',regKey2='dataProstate2D',regKey3='dataRectum2D')
    else:
        data_generator = Generator_2D_slices(path_patients_h5,batch_size,inputKey='dataMR2D',outputKey='dataSeg2D')
    data_generator_test = Generator_2D_slices(path_patients_h5_test,batch_size,inputKey='dataMR2D',outputKey='dataSeg2D')
    ####configs#####
    isDiceLoss = True # for dice loss
    isSoftmaxLoss = True #for softmax loss
    isAdLoss = True #for adverarail loss
    how2normalize = 3 #1. mu/(max-min); 2. mu/(percent_99 - percent_1); 3. mu/std
    prefixModelName = 'Segmentor_wdice_wce_model_1019_'
    prefixPredictedFN = 'preSub_wdice_wce_1019_'
    
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
    numofIters = 200000
    running_loss = 0.0
    start = time.time()
    for iter in range(numofIters):
        #print('iter %d'%iter)
        
        if isSegReg:
            inputs,labels, regGT1, regGT2, regGT3 = data_generator.next()
        else:
            inputs,labels = data_generator.next()

        labels = np.squeeze(labels)
        labels = labels.astype(int)

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        inputs = inputs.cuda()
        labels = labels.cuda()
        #we should consider different data to train
        
        #wrap them into Variable
        inputs,labels = Variable(inputs),Variable(labels)
        
        #zero the parameter gradients
        #netD.zero_grad()
        
        #forward + backward +optimizer    
        outputG = netG(inputs)
        outputG = softmax2d(outputG) #batach
#         print 'outputG: ',outputG.size(),'labels: ',labels.size()
#         print 'outputG: ', outputG.data[0].size()
        outputG = outputG.data.max(1)[1]
        #outputG = torch.squeeze(outputG) #[N,C,W,H]
        labels = labels.unsqueeze(1) #expand the 1st dim
#         print 'outputG: ',outputG.size(),'labels: ',labels.size()
        outputR = labels.type(torch.FloatTensor).cuda() #output_Real
        outputG = Variable(outputG.type(torch.FloatTensor).cuda())
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
        if isSegReg:
            outputG, outputReg1, outputReg2, outputReg3 = netG(inputs)
        else:
            outputG = netG(inputs) #here I am not sure whether we should use twice or not
        netG.zero_grad()
 
#         print '.............now test gradients function................................'
#         res = torch.autograd.gradcheck(myDiceLoss4Organ, (outputG,torch.squeeze(labels)))
#         
#         print res
#         print '.............gradients function check finished...........................' 
        
        if isSoftmaxLoss:
            lossG_G = criterion_CE2D(outputG,torch.squeeze(labels)) 
            lossG_G.backward(retain_graph=True) #compute gradients
        
#         criterion_dice(outputG,torch.squeeze(labels))
#         print 'hahaN'
        if isSegReg:
            lossG_Reg1 = criterion_MSE(outputReg1, regGT1)
            lossG_Reg2 = criterion_MSE(outputReg2, regGT2)
            lossG_Reg3 = criterion_MSE(outputReg3, regGT3)
            lossG_Reg = lossG_Reg1 + lossG_Reg2 + lossG_Reg3
            lossG_Reg.backward()

        if isDiceLoss:
#             criterion_dice = myWeightedDiceLoss4Organs(organIDs=[0,1,2,3], organWeights=[1,4,8,6])
            loss_dice = criterion_dice(outputG,torch.squeeze(labels))
#             loss_dice = myDiceLoss4Organs(outputG,torch.squeeze(labels)) #succeed
#             loss_dice.backward(retain_graph=True) #compute gradients for dice loss
            loss_dice.backward() #compute gradients for dice loss
        
        #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
        #angel of equation (note the max and min difference for generator and discriminator)
        outputG = netG(inputs)
        outputG = outputG.data.max(1)[1]
        outputG = Variable(outputG.type(torch.FloatTensor).cuda())
#         print 'outputG shape, ',outputG.size()

        lambda1 = 0
        outputD = netD(outputG.unsqueeze(1))
        lossG_D = lambda1 * criterion(outputD, real_label) #note, for generator, the label for outputG is real
        lossG_D.backward()
        
        #for other losses, we can define the loss function following the pytorch tutorial
        
        optimizerG.step() #update network parameters
#         print 'gradients of parameters****************************'
#         [x.grad.data for x in netG.parameters()]
#         print x.grad.data[0]
#         print '****************************'
        if isDiceLoss and isSoftmaxLoss and isAdLoss and isSegReg:
            lossG = lambda1 * lossG_D + lossG_G+loss_dice.data[0] + lossG_Reg
        elif isSoftmaxLoss and isAdLoss and isSegReg:
            lossG = lambda1 * lossG_D + lossG_G + lossG_Reg
        elif isDiceLoss and isAdLoss and isSegReg:
            lossG = lambda1 * lossG_D + loss_dice.data[0] + lossG_Reg    
        elif isDiceLoss and isSoftmaxLoss and isAdLoss:
            lossG = lambda1 * lossG_D + lossG_G + loss_dice.data[0]
        elif isSoftmaxLoss and isAdLoss:
            lossG = lambda1 * lossG_D + lossG_G
        elif isDiceLoss and isAdLoss:
            lossG = lambda1 * lossG_D + loss_dice.data[0]
        #print('loss for generator is %f'%lossG.data[0])
        #print statistics
        running_loss = running_loss + lossG.data[0]
#         print 'running_loss is ',running_loss,' type: ',type(running_loss)
        
#         print type(outputD_fake.cpu().data[0].numpy())
        
        if iter%100==0: #print every 2000 mini-batches
            print '************************************************'
            print 'time now is: ' + time.asctime(time.localtime(time.time()))
            print 'the outputD_real for iter {}'.format(iter), ' is ',outputD_real.cpu().data[0].numpy()[0]
            print 'the outputD_fake for iter {}'.format(iter), ' is ',outputD_fake.cpu().data[0].numpy()[0]
#             print 'running loss is ',running_loss
            print 'average running loss for generator between iter [%d, %d] is: %.3f'%(iter - 100 + 1,iter,running_loss/100)
            print 'loss for discriminator at iter ',iter, ' is %f'%lossD.data[0]
            print 'total loss for generator at iter ',iter, ' is %f'%lossG.data[0]
            if isDiceLoss and isSoftmaxLoss and isAdLoss and isSegReg:
                print 'lossG_D, lossG_G and loss_dice loss_Reg are %.2f, %.2f and %.2f respectively.'%(lossG_D.data[0], lossG_G.data[0], loss_dice.data[0], lossG_Reg.data[0])
            elif isDiceLoss and isSoftmaxLoss and isAdLoss:
                print 'lossG_D, lossG_G and loss_dice are %.2f, %.2f and %.2f respectively.'%(lossG_D.data[0], lossG_G.data[0], loss_dice.data[0])
            elif isSoftmaxLoss and isAdLoss:
                print 'lossG_D, lossG_G and loss_dice are %.2f and %.2f respectively.'%(lossG_D.data[0], lossG_G.data[0])
            elif isDiceLoss and isAdLoss:
                print 'lossG_D and loss_dice are %.2f and %.2f respectively.'%(lossG_D.data[0], loss_dice.data[0])

            print 'cost time for iter [%d, %d] is %.2f'%(iter - 100 + 1,iter, time.time()-start)
            print '************************************************'
            running_loss = 0.0
            start = time.time()
        if iter%2000==0: #save the model
        #if iter%300==299:
            torch.save(netG.state_dict(), prefixModelName+'%d.pt'%iter)
            print 'save model: '+prefixModelName+'%d.pt'%iter
            
        if iter%2000==0: #test one subject   

                mr_test_itk=sitk.ReadImage(os.path.join(path_test,'img1_nocrop.nii.gz'))
                ct_test_itk=sitk.ReadImage(os.path.join(path_test,'img1_label_nie_nocrop.nii.gz'))
                
                mrnp=sitk.GetArrayFromImage(mr_test_itk)
                mu=np.mean(mrnp)
                
                #for training data in pelvicSeg/pelvicH5
                #mrnp=(mrnp-mu)/(np.max(mrnp)-np.min(mrnp))
                ctnp=sitk.GetArrayFromImage(ct_test_itk)
                
                #for training data in pelvicSeg
                if how2normalize == 1:
                    maxV, minV=np.percentile(mrnp, [99 ,1])
                    print 'maxV,',maxV,' minV, ',minV
                    mrnp=(mrnp-mu)/(maxV-minV)
                    print 'unique value: ',np.unique(ctnp)

                #for training data in pelvicSeg
                if how2normalize == 2:
                    maxV, minV = np.percentile(mrnp, [99 ,1])
                    print 'maxV,',maxV,' minV, ',minV
                    mrnp = (mrnp-mu)/(maxV-minV)
                    print 'unique value: ',np.unique(ctnp)
                
                #for training data in pelvicSegRegH5
                if how2normalize== 3:
                    std = np.std(mrnp)
                    mrnp = (mrnp - mu)/std
                    print 'maxV,',np.ndarray.max(mrnp),' minV, ',np.ndarray.min(mrnp)
        
#                 print mrnp.dtype
#                 print ctnp.dtype
                
                #full image version with average over the overlapping regions
                #ct_estimated = testOneSubject(mrnp,ctnp,[3,168,112],[1,168,112],[1,8,8],netG,'Segmentor_model_%d.pt'%iter)

                # the attention regions
                x1=80
                x2=192
                y1=35
                y2=235
                matFA = mrnp[:,y1:y2,x1:x2] #note, matFA and matFAOut same size 
                matGT = ctnp[:,y1:y2,x1:x2]
#                 volFA = sitk.GetImageFromArray(matFA)
#                 sitk.WriteImage(volFA,'volFA'+'.nii.gz')
#                 volGT = sitk.GetImageFromArray(matGT)
#                 sitk.WriteImage(volGT,'volGT'+'.nii.gz')
                
                matOut,_ = testOneSubject(matFA,matGT,4,[3,168,112],[1,168,112],[1,8,8],ResSegNet(),prefixModelName+'%d.pt'%iter)
                ct_estimated = np.zeros([ctnp.shape[0],ctnp.shape[1],ctnp.shape[2]])
                print 'matOut shape: ',matOut.shape
                ct_estimated[:,y1:y2,x1:x2] = matOut
    
                ct_estimated = np.rint(ct_estimated) 
                diceBladder = dice(ct_estimated,ctnp,1)
                diceProstate = dice(ct_estimated,ctnp,2)
                diceRectumm = dice(ct_estimated,ctnp,3)
                
                print 'pred: ',ct_estimated.dtype, ' shape: ',ct_estimated.shape
                print 'gt: ',ctnp.dtype,' shape: ',ct_estimated.shape
                print 'dice1 = ',diceBladder,' dice2= ',diceProstate,' dice3= ',diceRectumm
                volout = sitk.GetImageFromArray(ct_estimated)
                sitk.WriteImage(volout,prefixPredictedFN+'{}'.format(iter)+'.nii.gz')    
            #             netG.save_state_dict('Segmentor_model_%d.pt'%iter)
#             netD.save_state_dic('Discriminator_model_%d.pt'%iter)
        
    print('Finished Training')
    
if __name__ == '__main__':
#     testGradients()     
    main()
    
