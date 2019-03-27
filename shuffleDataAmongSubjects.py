import numpy as np
import h5py
import os


'''
    Shuffle data (patches) among the subjects
input:
    save_dir: the h5 files you save
    num: the num of components in the h5 file
output:
    save as the same name h5 files
'''
def shuffleDataAmongSubjects(save_dir,savepath):
#     allfilenames = os.listdir(save_dir)
#     allfilenames = filter(lambda x: '.h5' in x and 'train' in x, allfilenames)
    
    nn = 7000
    dataMR = np.zeros([nn,1,3,168,112], dtype = np.float16)
    dataMR2D = np.zeros([nn,3,168,112], dtype = np.float16)
    dataProstate2D = np.zeros([nn,3,168,112], dtype = np.float16)
    dataRectum2D = np.zeros([nn,3,168,112], dtype = np.float16)
    dataBladder2D = np.zeros([nn,3,168,112], dtype = np.float16)
    dataSeg = np.zeros([nn,1,1,168,112], dtype = np.int8)
    dataSeg2D = np.zeros([nn,1,168,112], dtype = np.int8)
    dataContour = np.zeros([nn,1,1,168,112], dtype = np.int8)
    dataContour2D = np.zeros([nn,1,168,112], dtype = np.int8)
#                 

    allfilenames = os.listdir(save_dir)
#     print allfilenames
    allfilenames = filter(lambda x: '.h5' in x and 'train' in x, allfilenames)
#     print allfilenames
    cnt = 0
    numInOneSub = 5
    batchID = 0
    startInd = 0
    savefilename = 'train3x168x112_segregcontour_batch'
    for i_file, filename in enumerate(allfilenames):
        
        with h5py.File(os.path.join(save_dir, filename), 'r+') as h5f:
            print '*******path is ',os.path.join(save_dir,filename) 
            dMR = h5f['dataMR'][:]
            dMR2D = h5f['dataMR2D'][:]
            dProstate2D = h5f['dataProstate2D'][:]
            dRectum2D = h5f['dataRectum2D'][:]
            dBladder2D = h5f['dataBladder2D'][:]
            dSeg = h5f['dataSeg'][:]
            dSeg2D = h5f['dataSeg2D'][:]
            dContour = h5f['dataContour'][:]
            dContour2D = h5f['dataContour2D'][:]
            
            unitNum = dMR.shape[0]
            print 'unitNum: ',unitNum, 'dataMR shape: ',dataMR.shape
            dataMR[startInd:startInd + unitNum,...] = dMR
            dataMR2D[startInd : startInd + unitNum,...] = dMR2D
            dataProstate2D[startInd : startInd + unitNum,...] = dProstate2D
            dataRectum2D[startInd : startInd + unitNum,...] = dRectum2D
            dataBladder2D[startInd : startInd + unitNum,...] = dBladder2D
            dataSeg[startInd : startInd + unitNum,...] = dSeg
            dataSeg2D[startInd : startInd + unitNum,...] = dSeg2D
            dataContour[startInd : startInd + unitNum,...] = dContour
            dataContour2D[startInd : startInd + unitNum,...] = dContour2D
 
            
            startInd = startInd + unitNum
            
            cnt = cnt + 1
            
            if cnt == numInOneSub: 
                batchID = batchID + 1
                dataMR = dataMR[0:startInd,...]
                dataMR2D = dataMR2D[0:startInd,...]
                dataProstate2D = dataProstate2D[0:startInd,...]
                dataRectum2D = dataRectum2D[0:startInd,...]
                dataBladder2D = dataBladder2D[0:startInd,...]
                dataSeg = dataSeg[0:startInd,...]
                dataSeg2D = dataSeg2D[0:startInd,...]
                dataContour = dataContour[0:startInd,...]
                dataContour2D = dataContour2D[0:startInd,...]
                with h5py.File(os.path.join(savepath,savefilename+'{}.h5'.format(batchID)),'w') as hf:          
                    hf.create_dataset('dataMR', data=dataMR)
                    hf.create_dataset('dataMR2D', data=dataMR2D)
                    hf.create_dataset('dataProstate2D', data=dataProstate2D)
                    hf.create_dataset('dataRectum2D', data=dataRectum2D)
                    hf.create_dataset('dataBladder2D', data=dataBladder2D) 
                    hf.create_dataset('dataSeg', data=dataSeg)
                    hf.create_dataset('dataSeg2D', data=dataSeg2D)
                    hf.create_dataset('dataContour', data=dataContour)
                    hf.create_dataset('dataContour2D', data=dataContour2D)

                ############ initialization ###############
                cnt = 0
                startInd = 0
                print 'nn:', nn
                dataMR = np.zeros([nn,1,3,168,112], dtype = np.float16)
                dataMR2D = np.zeros([nn,3,168,112], dtype = np.float16)
                dataProstate2D = np.zeros([nn,3,168,112], dtype = np.float16)
                dataRectum2D = np.zeros([nn,3,168,112], dtype = np.float16)
                dataBladder2D = np.zeros([nn,3,168,112], dtype = np.float16)
                dataSeg = np.zeros([nn,1,1,168,112], dtype = np.int8)
                dataSeg2D = np.zeros([nn,1,168,112], dtype = np.int8)
                dataContour = np.zeros([nn,1,1,168,112], dtype = np.int8)
                dataContour2D = np.zeros([nn,1,168,112], dtype = np.int8)
            
#         mean_train, std_train = 0., 0
    batchID = batchID + 1
    if startInd!=0:
        dataMR = dataMR[0:startInd,...]
        dataMR2D = dataMR2D[0:startInd,...]
        dataProstate2D = dataProstate2D[0:startInd,...]
        dataRectum2D = dataRectum2D[0:startInd,...]
        dataBladder2D = dataBladder2D[0:startInd,...]
        dataSeg = dataSeg[0:startInd,...]
        dataSeg2D = dataSeg2D[0:startInd,...]
        dataContour = dataContour[0:startInd,...]
        dataContour2D = dataContour2D[0:startInd,...]
        with h5py.File(os.path.join(savepath,savefilename+'{}.h5'.format(batchID)),'w') as hf:          
            hf.create_dataset('dataMR', data=dataMR)
            hf.create_dataset('dataMR2D', data=dataMR2D)
            hf.create_dataset('dataProstate2D', data=dataProstate2D)
            hf.create_dataset('dataRectum2D', data=dataRectum2D)
            hf.create_dataset('dataBladder2D', data=dataBladder2D) 
            hf.create_dataset('dataSeg', data=dataSeg)
            hf.create_dataset('dataSeg2D', data=dataSeg2D)
            hf.create_dataset('dataContour', data=dataContour)
            hf.create_dataset('dataContour2D', data=dataContour2D)

    return

def main():
    path = '/shenlab/lab_stor5/dongnie/pelvic/pelvicSegRegContourH5/'
    savepath = '/shenlab/lab_stor5/dongnie/pelvic/pelvicSegRegContourBatchH5/'
    shuffleDataAmongSubjects(path,savepath)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
