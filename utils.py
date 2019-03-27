import numpy as np
import os
import SimpleITK as sitk
import h5py
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
import torch.nn.init
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import copy

#Dong add keys here
def Generator_2D_slices(path_patients,batchsize,inputKey='dataMR',outputKey='dataCT'):
    #path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)#every file  is a hdf5 patient
    while True:
        
        for idx,namepatient in enumerate(patients):
            print namepatient            
            f=h5py.File(os.path.join(path_patients,namepatient))
            #dataMRptr=f['dataMR']
            dataMRptr=f[inputKey]
            dataMR=dataMRptr.value
            
            #dataCTptr=f['dataCT']
            dataCTptr=f[outputKey]
            dataCT=dataCTptr.value

            dataMR=np.squeeze(dataMR)
            dataCT=np.squeeze(dataCT)

            #print 'mr shape h5 ',dataMR.shape#B,H,W,C
            #print 'ct shape h5 ',dataCT.shape#B,H,W
            
            shapedata=dataMR.shape
            #Shuffle data
            idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            dataMR=dataMR[idx_rnd,...]
            dataCT=dataCT[idx_rnd,...]
            modulo=np.mod(shapedata[0],batchsize)
################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0:
                to_add=batchsize-modulo
                inds_toadd=np.random.randint(0,dataMR.shape[0],to_add)
                X=np.zeros((dataMR.shape[0]+to_add,dataMR.shape[1],dataMR.shape[2],dataMR.shape[3]))#dataMR
                X[:dataMR.shape[0],...]=dataMR
                X[dataMR.shape[0]:,...]=dataMR[inds_toadd]                
                
                y=np.zeros((dataCT.shape[0]+to_add,dataCT.shape[1],dataCT.shape[2]))#dataCT
                y[:dataCT.shape[0],...]=dataCT
                y[dataCT.shape[0]:,...]=dataCT[inds_toadd]
                
            else:
                X=np.copy(dataMR)                
                y=np.copy(dataCT)

            #X = np.expand_dims(X, axis=3)    
            X=X.astype(np.float32)
            y=np.expand_dims(y, axis=3)#B,H,W,C
            y=y.astype(np.float32)
            #y[np.where(y==5)]=0
            
            #shuffle the data, by dong
            inds = np.arange(X.shape[0])
            np.random.shuffle(inds)
            X=X[inds,...]
            y=y[inds,...]
            
            print 'y shape ', y.shape                   
            for i_batch in xrange(int(X.shape[0]/batchsize)):
                yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...],  y[i_batch*batchsize:(i_batch+1)*batchsize,...])


#Dong add keys here
def Generator_2D_slicesV1(path_patients,batchsize,inputKey='dataMR',segKey='dataCT', contourKey='dataContour'):
    #path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)#every file  is a hdf5 patient
    while True:
        
        for idx,namepatient in enumerate(patients):
            print namepatient            
            f=h5py.File(os.path.join(path_patients,namepatient))
            #dataMRptr=f['dataMR']
            dataMRptr=f[inputKey]
            dataMR=dataMRptr.value
            
            #dataCTptr=f['dataCT']
            dataCTptr=f[segKey]
            dataCT=dataCTptr.value

            dataContourptr=f[contourKey]
            dataContour=dataContourptr.value

            dataMR=np.squeeze(dataMR)
            dataCT=np.squeeze(dataCT)
            dataContour=np.squeeze(dataContour)

            #print 'mr shape h5 ',dataMR.shape#B,H,W,C
            #print 'ct shape h5 ',dataCT.shape#B,H,W
            
            shapedata=dataMR.shape
            #Shuffle data
            idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            dataMR=dataMR[idx_rnd,...]
            dataCT=dataCT[idx_rnd,...]
            dataContour=dataContour[idx_rnd,...]
            modulo=np.mod(shapedata[0],batchsize)
################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0:
                to_add = batchsize-modulo
                inds_toadd = np.random.randint(0,dataMR.shape[0],to_add)
                X = np.zeros((dataMR.shape[0]+to_add, dataMR.shape[1], dataMR.shape[2], dataMR.shape[3]))#dataMR
                X[:dataMR.shape[0],...]=dataMR
                X[dataMR.shape[0]:,...]=dataMR[inds_toadd]                
                
                y=np.zeros((dataCT.shape[0]+to_add,dataCT.shape[1],dataCT.shape[2]))#dataCT
                y[:dataCT.shape[0],...]=dataCT
                y[dataCT.shape[0]:,...]=dataCT[inds_toadd]
                
                y1=np.zeros((dataContour.shape[0]+to_add,dataContour.shape[1],dataContour.shape[2]))#dataCT
                y1[:dataContour.shape[0],...]=dataContour
                y1[dataContour.shape[0]:,...]=dataContour[inds_toadd]
                
            else:
                X=np.copy(dataMR)                
                y=np.copy(dataCT)
                y1 = np.copy(dataContour)

            #X = np.expand_dims(X, axis=3)    
            X=X.astype(np.float32)
#             y=np.expand_dims(y, axis=3)#B,H,W,C
#             y=y.astype(np.float32)
            #y[np.where(y==5)]=0
            
            #shuffle the data, by dong
            inds = np.arange(X.shape[0])
            np.random.shuffle(inds)
            X=X[inds,...]
            y=y[inds,...]
            y1=y1[inds,...]
            
            print 'y shape ', y.shape                   
            for i_batch in xrange(int(X.shape[0]/batchsize)):
                yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...],  y[i_batch*batchsize:(i_batch+1)*batchsize,...],y1[i_batch*batchsize:(i_batch+1)*batchsize,...])
                

#Dong add a variable of keys here
'''
Input:
    path_patients: h5 data path
    batchsize: the batchsize we extract patches at a time
    keys: a variable number of parameters with a data structure of list
Output:
    The list of corresponding values indexed by the keys
'''
def Generator_2D_slices_variousKeys(path_patients,batchsize, keys):
    #path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)#every file  is a hdf5 patient
    numOfKeys = len(keys)
    while True:
        
        for idx,namepatient in enumerate(patients):
            print namepatient            
            f = h5py.File(os.path.join(path_patients,namepatient))
            
            data0 = f[keys[0]].value
            shapedata=data0.shape
#             idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            assert len(shapedata) == 4, 'data should have shape like: NxCxHxW'
            keyvalue = np.zeros((shapedata[0],shapedata[1],shapedata[2],shapedata[3],shapedata[4],numOfKeys))
            
            for keyInd in range(0,numOfKeys):
                #dataMRptr=f['dataMR']
                key = keys[keyInd]
                keyvalue[:,:,:,:,keyInd] = f[key].value
            
#                 dataMR=np.squeeze(dataMR)
#                 dataCT=np.squeeze(dataCT)
    
                #print 'mr shape h5 ',dataMR.shape#B,H,W,C
                #print 'ct shape h5 ',dataCT.shape#B,H,W
                
                
                #Shuffle data
            idx_rnd = np.random.choice(shapedata[0], shapedata[0], replace=False)
            keyvalue = keyvalue[idx_rnd,...]

            
            modulo = np.mod(shapedata[0],batchsize)
################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0: #we consider the remaining parts (e.g., 10008%8 = 2)
                to_add = batchsize-modulo
                inds_toadd = np.random.randint(0,keyvalue.shape[0],to_add)
                X = np.zeros((keyvalue.shape[0]+to_add,keyvalue.shape[1],keyvalue.shape[2],keyvalue.shape[3], keyvalue.shape[4], numOfKeys))#keyvalue
                X[:keyvalue.shape[0],...] = keyvalue
                X[keyvalue.shape[0]:,...] = keyvalue[inds_toadd]                
                
#                 y=np.zeros((dataCT.shape[0]+to_add,dataCT.shape[1],dataCT.shape[2]))#dataCT
#                 y[:dataCT.shape[0],...]=dataCT
#                 y[dataCT.shape[0]:,...]=dataCT[inds_toadd]
                
            else:
                X = np.copy(keyvalue)
#                 X=np.copy(dataMR)                
#                 y=np.copy(dataCT)

            #X = np.expand_dims(X, axis=3)    
            X=X.astype(np.float32)
#             y=np.expand_dims(y, axis=3)#B,H,W,C
#             y=y.astype(np.float32)
            #y[np.where(y==5)]=0
            
            #shuffle the data, by dong
            # inds = np.arange(X.shape[0])
            # np.random.shuffle(inds)
            # X=X[inds,...]
#             y=y[inds,...]
            
            print 'X shape ', X.shape                   
            for i_batch in xrange(int(X.shape[0]/batchsize)):
#                 yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...,keyInd],  y[i_batch*batchsize:(i_batch+1)*batchsize,...])
                yield ([ X[i_batch*batchsize:(i_batch+1)*batchsize,...,keyInd] for keyInd in range(0,numOfKeys)])


'''
 only consider the input without labeled data
'''
def Generator_3D_patches_unlabeled(path_patients, batchsize, inputKey='dataMR'):
#path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)#every file  is a hdf5 patient
    while True:
        
        for idx,namepatient in enumerate(patients):
            print namepatient            
            f=h5py.File(os.path.join(path_patients,namepatient))
            dataMRptr=f[inputKey]
            dataMR=dataMRptr.value
            #dataMR=np.squeeze(dataMR)
            
           
            #dataCT=np.squeeze(dataCT)

            dataMR=(dataMR)
            print 'mr shape h5 ',dataMR.shape

            
            shapedata=dataMR.shape
            #Shuffle data
            idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            dataMR=dataMR[idx_rnd,...]
            modulo=np.mod(shapedata[0],batchsize)
################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0:
                to_add=batchsize-modulo
                inds_toadd=np.random.randint(0, dataMR.shape[0], to_add)
                X=np.zeros((dataMR.shape[0]+to_add, dataMR.shape[1], dataMR.shape[2], dataMR.shape[3], dataMR.shape[4]))#dataMR
                X[:dataMR.shape[0],...]=dataMR
                X[dataMR.shape[0]:,...]=dataMR[inds_toadd]                
            else:
                X=np.copy(dataMR)

#             X = np.expand_dims(X, axis=4)     
            X=X.astype(np.float32)
            
            print 'X shape ', X.shape
                             
            for i_batch in xrange(int(X.shape[0]/batchsize)):
                yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...])
 
 
def Generator_3D_patches(path_patients, batchsize, inputKey='dataMR', outputKey='dataCT'):
    #path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)#every file  is a hdf5 patient
    while True:
        
        for idx,namepatient in enumerate(patients):
            print namepatient            
            f=h5py.File(os.path.join(path_patients,namepatient))
            dataMRptr=f[inputKey]
            dataMR=dataMRptr.value
            #dataMR=np.squeeze(dataMR)
            
            dataCTptr=f[outputKey]
            dataCT=dataCTptr.value
            #dataCT=np.squeeze(dataCT)

            dataMR=(dataMR)
            dataCT=(dataCT)
            print 'mr shape h5 ',dataMR.shape

            
            shapedata=dataMR.shape
            #Shuffle data
            idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            dataMR=dataMR[idx_rnd,...]
            dataCT=dataCT[idx_rnd,...]
            modulo=np.mod(shapedata[0],batchsize)
################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0:
                to_add=batchsize-modulo
                inds_toadd=np.random.randint(0, dataMR.shape[0], to_add)
                X=np.zeros((dataMR.shape[0]+to_add, dataMR.shape[1], dataMR.shape[2], dataMR.shape[3], dataMR.shape[4]))#dataMR
                X[:dataMR.shape[0],...]=dataMR
                X[dataMR.shape[0]:,...]=dataMR[inds_toadd]                
                
                y=np.zeros((dataCT.shape[0]+to_add, dataCT.shape[1], dataCT.shape[2], dataCT.shape[3], dataCT.shape[4]))#dataCT
                y[:dataCT.shape[0],...]=dataCT
                y[dataCT.shape[0]:,...]=dataCT[inds_toadd]
                
            else:
                X=np.copy(dataMR)                
                y=np.copy(dataCT)

#             X = np.expand_dims(X, axis=4)     
            X=X.astype(np.float32)
#             y=np.expand_dims(y, axis=4)
            y=y.astype(np.float32)
            
            print 'y shape ', y.shape
            print 'X shape ', X.shape
                             
            for i_batch in xrange(int(X.shape[0]/batchsize)):
                yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...],  y[i_batch*batchsize:(i_batch+1)*batchsize,...])


# Dong add a variable of keys here
'''
Input:
    path_patients: h5 data path
    batchsize: the batchsize we extract patches at a time
    keys: a variable number of parameters with a data structure of list
Output:
    The list of corresponding values indexed by the keys
'''
def Generator_3D_slices_variousKeys(path_patients, batchsize, keys):
    # path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)  # every file  is a hdf5 patient
    numOfKeys = len(keys)
    while True:

        for idx, namepatient in enumerate(patients):
            print namepatient
            f = h5py.File(os.path.join(path_patients, namepatient))

            data0 = f[keys[0]].value
            shapedata = data0.shape
            #             idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            assert len(shapedata) == 5, 'data should have shape like: NxCxHxWxD'
            keyvalue = np.zeros((shapedata[0], shapedata[1], shapedata[2], shapedata[3], shapedata[4], numOfKeys))

            for keyInd in range(0, numOfKeys):
                # dataMRptr=f['dataMR']
                key = keys[keyInd]
                keyvalue[:, :, :, :, :, keyInd] = f[key].value

            #                 dataMR=np.squeeze(dataMR)
            #                 dataCT=np.squeeze(dataCT)

            # print 'mr shape h5 ',dataMR.shape#B,H,W,C
            # print 'ct shape h5 ',dataCT.shape#B,H,W

            # Shuffle data
            idx_rnd = np.random.choice(shapedata[0], shapedata[0], replace=False)
            keyvalue = keyvalue[idx_rnd, ...]

            modulo = np.mod(shapedata[0], batchsize)
            ################## always the number of samples will be a multiple of batchsz##########################3
            if modulo != 0:  # we consider the remaining parts (e.g., 10008%8 = 2)
                to_add = batchsize - modulo
                inds_toadd = np.random.randint(0, keyvalue.shape[0], to_add)
                X = np.zeros((keyvalue.shape[0] + to_add, keyvalue.shape[1], keyvalue.shape[2], keyvalue.shape[3],
                              keyvalue.shape[4], numOfKeys))  # keyvalue
                X[:keyvalue.shape[0], ...] = keyvalue
                X[keyvalue.shape[0]:, ...] = keyvalue[inds_toadd]

            #                 y=np.zeros((dataCT.shape[0]+to_add,dataCT.shape[1],dataCT.shape[2]))#dataCT
            #                 y[:dataCT.shape[0],...]=dataCT
            #                 y[dataCT.shape[0]:,...]=dataCT[inds_toadd]

            else:
                X = np.copy(keyvalue)
            #                 X=np.copy(dataMR)
            #                 y=np.copy(dataCT)

            # X = np.expand_dims(X, axis=3)
            X = X.astype(np.float32)
            #             y=np.expand_dims(y, axis=3)#B,H,W,C
            #             y=y.astype(np.float32)
            # y[np.where(y==5)]=0

            # shuffle the data, by dong
            # inds = np.arange(X.shape[0])
            # np.random.shuffle(inds)
            # X = X[inds, ...]
            #             y=y[inds,...]

            print 'X shape ', X.shape
            for i_batch in xrange(int(X.shape[0] / batchsize)):
                #                 yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...,keyInd],  y[i_batch*batchsize:(i_batch+1)*batchsize,...])
                yield ([X[i_batch * batchsize:(i_batch + 1) * batchsize, ..., keyInd] for keyInd in range(0, numOfKeys)])


'''
    custom weights initialization called on netG and netD
    I think this can only goes into the 1st space, instead of recursive initialization
 '''
def weights_init(m):
    xavier=torch.nn.init.xavier_uniform
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
        #print m.weight.data
        #print m.bias.data
        xavier(m.weight.data)
#         print 'come xavier'
        #xavier(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear')!=-1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

'''
    this function is used to compute the dice ratio
input:
    im1: gt
    im2 pred
    tid: the id for consideration
output:
    dcs
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

'''
    this function is used to compute the intersection over Union
input:
    im1: gt
    im2 pred
    tid: the id for consideration
output:
    iou
'''
def IoU(im1, im2,tid):
    im1=im1==tid #make it boolean
    im2=im2==tid #make it boolean
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)
    
    eps =  1e-7

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    union = np.logical_or(im1, im2)
    iou = 1. * intersection.sum() / (union.sum()+eps)
    return iou


'''
    for finetune or sth else to transfer the weights from other models
'''
def transfer_weights(model_from, model_to):
    wf = copy.deepcopy(model_from.state_dict())
    wt = model_to.state_dict()
    for k in wt.keys():
        if not k in wf:
            wf[k] = wt[k]
    model_to.load_state_dict(wf)



'''
    Evaluate one patch using the latest pytorch model
input:
    patch_MR: a np array of shape [H,W,nchans]
output:    
    patch_CT_pred: segmentation maps for the corresponding input patch 
'''
def evaluate_oldversion(patch_MR, netG, modelPath):
    
    
        patch_MR = torch.from_numpy(patch_MR)

        patch_MR = patch_MR.unsqueeze(0)
#         patch_MR=np.expand_dims(patch_MR,axis=0)#[1,nchans,H,W]
        patch_MR = Variable(patch_MR).float().cuda()
#         netG = ResSegNet() #here is your network
#         checkpoint = torch.load(modelPath)
#         netG.load_state_dict(checkpoint['model'])
#         netG.load_state_dict(torch.load(modelPath)) #this is for old version
        netG.cuda()
        netG.eval()
#         print type(patch_MR)
        res = netG(patch_MR)
        
#         print res.size(),res.squeeze(0).size()
        if isinstance(res, tuple):
            res = res[0]
        _, tmp = res.squeeze(0).max(0)
        patchOut = tmp.data.cpu().numpy().squeeze()

        #imsave('mr32.png',np.squeeze(MR16_eval[0,:,:,2]))
        #imsave('ctpred.png',np.squeeze(patch_CT_pred[0,:,:,0]))
        #print 'mean of layer  ',np.mean(MR16_eval)
        #print 'min ct estimated ',np.min(patch_CT_pred)
        #print 'max ct estimated ',np.max(patch_CT_pred)
        #print 'mean of ctpatch estimated ',np.mean(patch_CT_pred)
        del tmp
        del patch_MR
        del _
        return patchOut

'''
    Evaluate one patch using the latest pytorch model
input:
    patch_MR: a np array of shape [H,W,nchans]
output:    
    patch_CT_pred: probability maps for the corresponding input patch 
    
'''
def evaluate(patch_MR, netG, modelPath):
    
    
        patch_MR = torch.from_numpy(patch_MR)

        patch_MR = patch_MR.unsqueeze(0)
#         patch_MR=np.expand_dims(patch_MR,axis=0)#[1,H,W,nchans]
        patch_MR = Variable(patch_MR).float().cuda()
#         netG = ResSegNet() #here is your network
#         netG.load_state_dict(torch.load(modelPath))
        netG.cuda()
        netG.eval()
#         print type(patch_MR)
        res = netG(patch_MR)
        
#         print res.size(),res.squeeze(0).size()
        if isinstance(res, tuple) or isinstance(res,list):
            res = res[0]
#         _, tmp = res.squeeze(0).max(0)
#         patchOut = tmp.data.cpu().numpy().squeeze()
        patchOut = res
        #patchOut = res.squeeze(0).data.cpu().numpy() #NxCxWxH

        #imsave('mr32.png',np.squeeze(MR16_eval[0,:,:,2]))
        #imsave('ctpred.png',np.squeeze(patch_CT_pred[0,:,:,0]))
        #print 'mean of layer  ',np.mean(MR16_eval)
        #print 'min ct estimated ',np.min(patch_CT_pred)
        #print 'max ct estimated ',np.max(patch_CT_pred)
        #print 'mean of ctpatch estimated ',np.mean(patch_CT_pred)
        return patchOut

'''
    Receives an MR image and returns an segmentation label maps with the same size
    We use averaging at the overlapping regions
input:
    MR_image: the raw input data (after preprocessing)
    CT_GT: the ground truth data
    MR_patch_sz: 3x168x112?
    CT_patch_sz: 1x168x112?
    step: 1 (along the 1st dimension)
    netG: network for the generator
    modelPath: the pytorch model path (pth)
output:
    matOut: the predicted segmentation map
'''
def testOneSubject(MR_image,CT_GT,MR_patch_sz,CT_patch_sz,step, netG, modelPath):

    matFA = MR_image
    matSeg = CT_GT
    dFA = MR_patch_sz
    dSeg = CT_patch_sz
    
    eps = 1e-5
    [row,col,leng] = matFA.shape
    margin1 = int((dFA[0]-dSeg[0])/2)
    margin2 = int((dFA[1]-dSeg[1])/2)
    margin3 = int((dFA[2]-dSeg[2])/2)
    cubicCnt = 0
    marginD = [margin1,margin2,margin3]
    print 'matFA shape is ',matFA.shape
    matFAOut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA
    
    matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA[0:marginD[0],:,:] #we'd better flip it along the first dimension
    matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA[row-marginD[0]:matFA.shape[0],:,:] #we'd better flip it along the 1st dimension
    
    matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]] = matFA[:,0:marginD[1],:] #we'd better flip it along the 2nd dimension
    matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]] = matFA[:,col-marginD[1]:matFA.shape[1],:] #we'd better to flip it along the 2nd dimension
    
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]] = matFA[:,:,0:marginD[2]] #we'd better flip it along the 3rd dimension
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]] = matFA[:,:,leng-marginD[2]:matFA.shape[2]]
    
    
    matOut = np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))
    used = np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))+eps
    #fid=open('trainxxx_list.txt','a');
    print 'last i ',row-dSeg[0]
    for i in range(0,row-dSeg[0]+1,step[0]):
#         print 'i ',i
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                #print 'volSeg shape is ',volSeg.shape
                volFA = matFAOut[i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                #print 'volFA shape is ',volFA.shape
                #mynet.blobs['dataMR'].data[0,0,...]=volFA
                #mynet.forward()
                #temppremat = mynet.blobs['softmax'].data[0].argmax(axis=0) #Note you have add softmax layer in deploy prototxt
                temppremat = evaluate(volFA, netG, modelPath)
                temppremat = temppremat.squeeze(0).data.cpu().numpy()
                temppremat = temppremat.argmax(axis=0).squeeze()
                
                #temppremat = evaluate(volFA, netG, modelPath)
#                 print 'temppremat shape 1: ',temppremat.shape
                if len(temppremat.shape)==2:
                    temppremat = np.expand_dims(temppremat,axis=0)
                #print 'patchout shape ',temppremat.shape
                #temppremat=volSeg
#                 print 'temppremat shape 2: ',temppremat.shape
                matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+temppremat;
                used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+1;
    matOut = matOut/used
    return matOut


'''
    Receives an MR image and returns an segmentation label maps with the same size
    We use majority voting at the overlapping regions
input:
    MR_image: the raw input data (after preprocessing)
    CT_GT: the ground truth data
    NumOfClass: number of classes
    MR_patch_sz: 3x168x112?
    CT_patch_sz: 1x168x112?
    step: 1 (along the 1st dimension)
    netG: network for the generator
    modelPath: the pytorch model path (pth)
output:
    matOut: the predicted segmentation map
'''
def testOneSubject(MR_image, CT_GT, NumOfClass, MR_patch_sz, CT_patch_sz, step, netG, modelPath, nd=2):
    eps=1e-5
    
    matFA = MR_image
    matSeg = CT_GT
    dFA = MR_patch_sz
    dSeg = CT_patch_sz
    
    [row,col,leng] = matFA.shape
    margin1 = (dFA[0]-dSeg[0])/2
    margin2 = (dFA[1]-dSeg[1])/2
    margin3 = (dFA[2]-dSeg[2])/2
    cubicCnt = 0
    marginD = [margin1,margin2,margin3]
    
    #print 'matFA shape is ',matFA.shape
    matFAOut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    #print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA

#     matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[0:marginD[0],:,:] #we'd better flip it along the first dimension
#     matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[row-marginD[0]:matFA.shape[0],:,:] #we'd better flip it along the 1st dimension
# 
#     matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,0:marginD[1],:] #we'd better flip it along the 2nd dimension
#     matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,col-marginD[1]:matFA.shape[1],:] #we'd better to flip it along the 2nd dimension
# 
#     matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,0:marginD[2]] #we'd better flip it along the 3rd dimension
#     matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,leng-marginD[2]:matFA.shape[2]]
    if margin1!=0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
    

#     dim1=np.arange(80,192)
#     dim2=np.arange(35,235)
#     x1=80
#     x2=192
#     y1=35
#     y2=235
# #     matFAOutScale = matFAOut[:,y1:y2,x1:x2] #note, matFA and matFAOut same size 
# #     matSegScale = matSeg[:,y1:y2,x1:x2]
    matFAOutScale = matFAOut
    matSegScale = matSeg
    matOut = np.zeros((matSegScale.shape[0],matSegScale.shape[1],matSegScale.shape[2],NumOfClass),dtype=np.int32)
    [row,col,leng] = matSegScale.shape
        
    cnt = 0
    #fid=open('trainxxx_list.txt','a');
    for i in range(0,row-dSeg[0]+1,step[0]):
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                #print 'volSeg shape is ',volSeg.shape
                volFA = matFAOutScale[i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                cnt = cnt + 1
                if nd==3:
                    volFA = np.expand_dims(volFA, axis=0)
                temppremat = evaluate(volFA, netG, modelPath)
                temppremat = temppremat.squeeze(0).data.cpu().numpy()
                temppremat = temppremat.argmax(axis=0).squeeze() #CxWxH->WxH
#                 volPre = sitk.GetImageFromArray(temppremat)
#                 sitk.WriteImage(volPre,'volPre_{}'.format(cnt)+'.nii.gz')
                
                if len(temppremat.shape)==2:
                    temppremat = np.expand_dims(temppremat,axis=0)
                for labelInd in range(NumOfClass): #note, start from 0
                    currLabelMat = np.where(temppremat==labelInd, 1, 0) # true, vote for 1, otherwise 0
                    #scio.savemat('volOut_%d'%cnt+'_label%d.mat'%labelInd,{'currLabelMat%d'%labelInd:currLabelMat})
                    matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2],labelInd] = matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2],labelInd]+currLabelMat;
       
    #scio.savemat('matOut%s.mat'%fileID,{'matOut':matOut})
    matOut = matOut.argmax(axis=3)
    matOut = np.rint(matOut) #this is necessary to convert the data type to be accepted by NIFTI, otherwise will appear strange errors
#     print 'line 378: matOut shape: ',matOut.shape
#     matOut1 = matOut
#     matOut1=np.zeros([matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]])
#     matOut1[:,y1:y2,x1:x2]=matOut
    #matOut1=np.transpose(matOut1,(2,1,0))
    #matSeg=np.transpose(matSeg,(2,1,0))
    return matOut,matSeg


'''
    Receives an MR image and returns an segmentation label maps with the same size
    We use majority voting at the overlapping regions
input:
    MR_image: the raw input data (after preprocessing)
    CT_GT: the ground truth data
    NumOfClass: number of classes
    MR_patch_sz: 3x168x112?
    CT_patch_sz: 1x168x112?
    step: 1 (along the 1st dimension)
    netG: network for the generator
    modelPath: the pytorch model path (pth)
    resType: 0: segmentation map (integer); 1: regression map (continuous); 2: segmentation map + probability map
output:
    matOut: the predicted segmentation map (or regression map or the probability map)
'''
def testOneSubject(MR_image, CT_GT, NumOfClass, MR_patch_sz, CT_patch_sz, step, netG, modelPath,resType=0,nd=2):
    eps=1e-5
    
    matFA = MR_image
    matSeg = CT_GT
    dFA = MR_patch_sz
    dSeg = CT_patch_sz
    
    if matFA.ndim==4:
        [ch, row,col,leng] = matFA.shape
    else:
        [row,col,leng] = matFA.shape
    margin1 = (dFA[0]-dSeg[0])/2
    margin2 = (dFA[1]-dSeg[1])/2
    margin3 = (dFA[2]-dSeg[2])/2
    cubicCnt = 0
    marginD = [margin1,margin2,margin3]
    
    #print 'matFA shape is ',matFA.shape


    if matFA.ndim==3:
        matFAOut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
        #print 'matFAOut shape is ',matFAOut.shape
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA
        if margin1!=0:
            matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
            matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
        if margin2!=0:
            matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
            matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
        if margin3!=0:
            matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
            matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
    else:
        matFAOut = matFA   
    matFAOutScale = matFAOut
    matSegScale = matSeg
#     print 'matFA.shape: ',matFA.shape

    matOut = np.zeros((matSegScale.shape[0],matSegScale.shape[1],matSegScale.shape[2],NumOfClass),dtype=np.int32)
    matProb = np.zeros((NumOfClass, matSeg.shape[0], matSeg.shape[1], matSeg.shape[2]))
    used = np.zeros((NumOfClass, matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))+eps
#     print 'matProb.shape: ',matProb.shape,' matOut.shape:', matOut.shape,' used.shape: ',used.shape
    [row,col,leng] = matSegScale.shape
    softmax2d = nn.Softmax2d()
    cnt = 0
    #fid=open('trainxxx_list.txt','a');
    for i in range(0,row-dSeg[0]+1,step[0]):
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                #print 'volSeg shape is ',volSeg.shape
                if matFA.ndim==3:
                    volFA = matFAOutScale[i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                else:
                    volFA = matFAOutScale[:,i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                cnt = cnt + 1
                if nd==3 and volFA.ndim==3:
                    volFA = np.expand_dims(volFA, axis=0)
#                 print 'volFA.shape: ',volFA.shape
                tempprobmat = evaluate(volFA, netG, modelPath)#NxCxWxH
#                 tempprobmat = tempprobmat.data.cpu().numpy() #NxCxWxH
#                 volPre = sitk.GetImageFromArray(temppremat)
#                 sitk.WriteImage(volPre,'volPre_{}'.format(cnt)+'.nii.gz')
                if resType==0 or resType==2:
                    tempprobmat = F.softmax(tempprobmat, dim=1)
                    tempprobmat = tempprobmat.squeeze(0).data.cpu().numpy()
                    temppremat = tempprobmat.argmax(axis=0).squeeze() #CxWxH->WxH
                    
                    if len(temppremat.shape)==2:
                        temppremat = np.expand_dims(temppremat,axis=0)
                    for labelInd in range(NumOfClass): #note, start from 0
                        currLabelMat = np.where(temppremat==labelInd, 1, 0) # true, vote for 1, otherwise 0
                        #scio.savemat('volOut_%d'%cnt+'_label%d.mat'%labelInd,{'currLabelMat%d'%labelInd:currLabelMat})
                        matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2],labelInd] = matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2],labelInd]+currLabelMat;
                if resType==1 or resType==2:
                    temppremat = tempprobmat.squeeze() #CxWxHx1->CxWxH
                    #print 'temppremat shape 1: ',temppremat.shape
                    if len(temppremat.shape)==3:#CxWxH->Cx1xHxW
                        temppremat = np.expand_dims(temppremat,axis=1)
                    #print 'patchout shape ',temppremat.shape
                    #temppremat=volSeg
#                     print 'temppremat.shape 2: ',temppremat.shape
                    matProb[:, i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = matProb[:, i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+temppremat
                    used[:, i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = used[:, i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+1
                            
    #scio.savemat('matOut%s.mat'%fileID,{'matOut':matOut})
    matOut = matOut.argmax(axis=3)
    matOut = np.rint(matOut) #this is necessary to convert the data type to be accepted by NIFTI, otherwise will appear strange errors
#     print 'matProb.max: ',np.ndarray.max(matProb),' matProb.min: ',np.ndarray.min(matProb),' used.max: ',np.ndarray.max(used),' used.min: ',np.ndarray.min(used)
    matProb = matProb/used
#     print 'line 378: matOut shape: ',matOut.shape
#     matOut1 = matOut
#     matOut1=np.zeros([matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]])
#     matOut1[:,y1:y2,x1:x2]=matOut
    #matOut1=np.transpose(matOut1,(2,1,0))
    #matSeg=np.transpose(matSeg,(2,1,0))
    if resType==0:
        return matOut,matSeg
    elif resType==1:
        return matProb,matSeg
    else:
        return matOut, matProb, matSeg




'''
    Receives an MR image and returns an segmentation label maps with the same size
    We use majority voting at the overlapping regions
input:
    MR_image: the raw input data (after preprocessing)
    CT_GT: the ground truth data
    NumOfClass: number of classes
    MR_patch_sz: 3x168x112?
    CT_patch_sz: 1x168x112?
    step: 1 (along the 1st dimension)
    netG: network for the generator
    modelPath: the pytorch model path (pth)
    resType: 0: segmentation map (integer); 1: regression map (continuous); 2: segmentation map + probability map
output:
    matOut: the predicted segmentation map (or regression map or the probability map)
'''
def testOneSubjectWith4DInput(MR_image, CT_GT, NumOfClass, MR_patch_sz, CT_patch_sz, step, netG, modelPath,resType=0,nd=2):
    eps=1e-5
    
    matFA = MR_image
    matSeg = CT_GT
    dFA = MR_patch_sz
    dSeg = CT_patch_sz
    
    [chn, row,col,leng] = matFA.shape
    margin1 = (dFA[0]-dSeg[0])/2
    margin2 = (dFA[1]-dSeg[1])/2
    margin3 = (dFA[2]-dSeg[2])/2
    cubicCnt = 0
    marginD = [margin1,margin2,margin3]
    
   
    matFAOutScale = matFA
    matSegScale = matSeg
    matOut = np.zeros((row,col,leng,NumOfClass),dtype=np.int32)
    matProb = np.zeros((NumOfClass, row,col,leng))
    used = np.zeros((NumOfClass, row,col,leng))+eps
#     print 'matProb.shape: ',matProb.shape,' matOut.shape:', matOut.shape,' used.shape: ',used.shape
    [chn, row,col,leng] = matSegScale.shape
    softmax2d = nn.Softmax2d()
    cnt = 0
    #fid=open('trainxxx_list.txt','a');
    for i in range(0,row-dSeg[0]+1,step[0]):
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                #print 'volSeg shape is ',volSeg.shape
                volFA = matFAOutScale[:,i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                cnt = cnt + 1
#                 if nd==3:
#                     volFA = np.expand_dims(volFA, axis=0)
                tempprobmat = evaluate(volFA, netG, modelPath)#NxCxWxH
#                 tempprobmat = tempprobmat.data.cpu().numpy() #NxCxWxH
#                 volPre = sitk.GetImageFromArray(temppremat)
#                 sitk.WriteImage(volPre,'volPre_{}'.format(cnt)+'.nii.gz')
                if resType==0 or resType==2:
                    tempprobmat = F.softmax(tempprobmat, dim=1)
                    tempprobmat = tempprobmat.squeeze(0).data.cpu().numpy()
                    temppremat = tempprobmat.argmax(axis=0).squeeze() #CxWxH->WxH
                    
                    if len(temppremat.shape)==2:
                        temppremat = np.expand_dims(temppremat,axis=0)
                    for labelInd in range(NumOfClass): #note, start from 0
                        currLabelMat = np.where(temppremat==labelInd, 1, 0) # true, vote for 1, otherwise 0
                        #scio.savemat('volOut_%d'%cnt+'_label%d.mat'%labelInd,{'currLabelMat%d'%labelInd:currLabelMat})
                        matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2],labelInd] = matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2],labelInd]+currLabelMat;
                if resType==1 or resType==2:
                    temppremat = tempprobmat.squeeze() #CxWxHx1->CxWxH
                    #print 'temppremat shape 1: ',temppremat.shape
                    if len(temppremat.shape)==3:#CxWxH->Cx1xHxW
                        temppremat = np.expand_dims(temppremat,axis=1)
                    #print 'patchout shape ',temppremat.shape
                    #temppremat=volSeg
#                     print 'temppremat.shape 2: ',temppremat.shape
                    matProb[:, i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = matProb[:, i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+temppremat
                    used[:, i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = used[:, i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+1
                            
    #scio.savemat('matOut%s.mat'%fileID,{'matOut':matOut})
    matOut = matOut.argmax(axis=3)
    matOut = np.rint(matOut) #this is necessary to convert the data type to be accepted by NIFTI, otherwise will appear strange errors
#     print 'matProb.max: ',np.ndarray.max(matProb),' matProb.min: ',np.ndarray.min(matProb),' used.max: ',np.ndarray.max(used),' used.min: ',np.ndarray.min(used)
    matProb = matProb/used
#     print 'line 378: matOut shape: ',matOut.shape
#     matOut1 = matOut
#     matOut1=np.zeros([matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]])
#     matOut1[:,y1:y2,x1:x2]=matOut
    #matOut1=np.transpose(matOut1,(2,1,0))
    #matSeg=np.transpose(matSeg,(2,1,0))
    if resType==0:
        return matOut,matSeg
    elif resType==1:
        return matProb,matSeg
    else:
        return matOut, matProb, matSeg




'''
    used as list in argparse
'''
def arg_as_list(s):                                                            
    v = ast.literal_eval(s)                                                    
    if type(v) is not list:                                                    
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v  

'''
    Takes a Tensor of size 1xHxW and create one-hot encoding of size nclassxHxW
    OneHotEncoding: [2,1,3,0]->[0 0 1 0; 0 1 0 0; 0 0 0 1; 1 0 0 0]
    label: NxWxHxD or NxWxH
'''
class OneHotEncode(object):
    def __init__(self, nclass=4, nd=2):
        self.nclass = nclass
        self.nd = nd
    def __call__(self,label):
#         label_a = np.array(transforms.ToPILImage()(label.byte().unsqueeze(0)),np.uint8)
        label_a = label.numpy().astype(np.uint8)
        if self.nd == 2:
            ohlabel = np.zeros((label_a.shape[0],self.nclass,label_a.shape[1],label_a.shape[2])).astype(np.uint8)
    
            for c in range(self.nclass):
                ohlabel[:,c, :,:] = (label_a == c).astype(np.uint8)
        elif self.nd==3:
            ohlabel = np.zeros((label_a.shape[0],self.nclass,label_a.shape[1],label_a.shape[2],label_a.shape[2])).astype(np.uint8)
    
            for c in range(self.nclass):
                ohlabel[:,c, :,:,:] = (label_a == c).astype(np.uint8)
        else:
            ohlabel = np.zeros((label_a.shape[0],self.nclass,label_a.shape[1],label_a.shape[2])).astype(np.uint8)
            print 'NDim should be 2 or 3'

        return torch.from_numpy(ohlabel)
    
'''
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
output: image + image*noise
'''
def noisy(noise_typ,image):
    if noise_typ == "gauss":
       row,col,ch= image.shape
       mean = 0
       var = 0.1
       sigma = var**0.5
       gauss = np.random.normal(mean,sigma,(row,col,ch))
       gauss = gauss.reshape(row,col,ch)
       noisy = image + gauss
       return noisy
    elif noise_typ == "s&p":
       row,col,ch = image.shape
       s_vs_p = 0.5
       amount = 0.004
       out = np.copy(image)
       # Salt mode
       num_salt = np.ceil(amount * image.size * s_vs_p)
       coords = [np.random.randint(0, i - 1, int(num_salt))
               for i in image.shape]
       out[coords] = 1
    
       # Pepper mode
       num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
       coords = [np.random.randint(0, i - 1, int(num_pepper))
               for i in image.shape]
       out[coords] = 0
       return out
    elif noise_typ == "poisson":
       vals = len(np.unique(image))
       vals = 2 ** np.ceil(np.log2(vals))
       noisy = np.random.poisson(image * vals) / float(vals)
       return noisy
    elif noise_typ =="speckle":
       row,col,ch = image.shape
       gauss = np.random.randn(row,col,ch)
       gauss = gauss.reshape(row,col,ch)        
       noisy = image + image * gauss
       
    return noisy

'''
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    'gauss'     Gaussian-distributed additive noise.

output: image + noise
'''
def addGaussianNoise(image,mean=0,var=0.01):
      row,col,ch= image.shape
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noiseImg = image + gauss
      return noiseImg
    