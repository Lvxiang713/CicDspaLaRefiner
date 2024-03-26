# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:51:30 2023
@author:LvXiang, JingRunyu
"""
import torch
import os, re, copy
import numpy as np
import chardet
from sklearn.impute import KNNImputer


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, fastaFileList,ignoreColNum=0):
        '''
        fastaFileList: A  list like:
            [file1, file2, ..., filen ]
            
        dataType: {dna, rna, protein, naivesmile}
        '''
        self.dataList = [] #(idx, filename, label, arr)
        self.labelDict = None
        self.classNum = None
        self.labelSet = set()
        self.indexCount = 0
        for f in fastaFileList:
            l = os.path.split(os.path.split(f)[0])[1]
            # self.fileList.append(f)
            # self.labelList.append(l)
            self.readFile(f,l,ignoreColNum=ignoreColNum)
            self.labelSet.add(l)
        self.queryIndex = np.arange(len(self.dataList),dtype=int)
        
        # self.generateLabelDict()
        self.classIndexDict = {}
        
        for dataIter in self.dataList:
            i = dataIter[0]
            l = dataIter[2]
            if not l in self.classIndexDict:
                self.classIndexDict[l] = []
            self.classIndexDict[l].append(i)
        
        self.feaOneHotBins = None
        self.labelOneHotDict = {}
    
    def generateLabelDict(self,force=False):
        '''
        deprecated
        '''
        if not self.labelDict is None:
            if not force:
                print('LabelDict is already generated')
                return
        labelCount = 0
        self.labelDict = {}
        for k in self.labelSet:
            self.labelDict[k] = labelCount
            labelCount += 1
    
    def shuffle(self,shuffleDict=None):
        # for l in self.classIndexDict:
        #     np.random.shuffle(self.classIndexDict[l])
        if shuffleDict is None:
            shuffleDict = {}
            for l in self.classIndexDict:
                subIndex = np.arange(len(self.classIndexDict[l]))
                np.random.shuffle(subIndex)
                shuffleDict[l] = subIndex
        
        for l in self.classIndexDict:
           self.classIndexDict[l] = list(np.array(self.classIndexDict[l])[shuffleDict[l]]) 
        return shuffleDict
    
    # def regenerateQueryIndex(self):
    #     self.queryIndex = []
    #     for l in self.classIndexDict:
    #         self.queryIndex += list(self.classIndexDict[l])
    #     self.labelOneHotDict = {}
        
    def regenerateQueryIndex(self):
        self.labelOneHotDict = {}
        newDataList = []
        newClassIndexDict = {}
        newQueryIndex = []
        queryNum = 0
        for l in self.classIndexDict:
            if not l in newClassIndexDict:
                newClassIndexDict[l] = []
            for i in  self.classIndexDict[l]:
                
                newClassIndexDict[l].append(queryNum)
                newQueryIndex.append(queryNum)
                
                tmpData = self.dataList[i]
                newDataList.append(tmpData)
                queryNum += 1
        self.dataList = newDataList
        self.classIndexDict = newClassIndexDict
        self.queryIndex = newQueryIndex
        
    def readFile(self,fileName,label,sep=',',ignoreColNum=0):
        
        with open(fileName, 'rb') as f:
            data = f.read(1000)
            result = chardet.detect(data)
            encoding = result['encoding']

        with open(fileName, encoding=encoding) as FID:
            for l in FID:
                if l.startswith('#'):
                    continue
                
                eles = re.sub('\s','',l.strip()).split(sep)
                tmparr = np.array(eles[ignoreColNum:],dtype = str)
                rowMask = []
                rowEle = []
                for ele in tmparr:
                    if ele == '' or ele.strip() == '':
                        rowEle.append('0')
                        rowMask.append(1)
                    else:
                        rowEle.append(ele)
                        rowMask.append(0)
                
                # tmparr[tmparr==''] = '0'
                arr = torch.tensor(np.array(rowEle, dtype=float))
                missMask = torch.tensor(np.array(rowMask, dtype=bool))
                # try:
                #     arr = torch.tensor(tmparr.astype(float))
                # except:
                #     print(arr)
                #     assert False
                self.dataList.append((self.indexCount,fileName,label,arr,missMask) )
                self.indexCount += 1
               

    def splitForVal(self, scale=0.2, shuffleDict=None):
        shuffleDict = self.shuffle(shuffleDict=shuffleDict)        
        outDataSet = copy.deepcopy(self)
        for l in self.classIndexDict:
            splitNum = int(len(self.classIndexDict[l]) * scale)
            self.classIndexDict[l] = self.classIndexDict[l][splitNum:]
            outDataSet.classIndexDict[l] = outDataSet.classIndexDict[l][:splitNum]
        self.regenerateQueryIndex()
        outDataSet.regenerateQueryIndex()
        
        return outDataSet,shuffleDict
    
    def generateSubDataset(self, scale=0.2, shuffleDict=None, method='noRepeat'):
        if method == 'bootstrap':
            return self.generateSubDatasetBootstrap(scale=scale, shuffleDict=shuffleDict)
        return self.generateSubDatasetNoRepeat(scale=scale, shuffleDict=shuffleDict)
    
    def generateSubDatasetBootstrap(self, scale=0.2, shuffleDict=None):
        shuffleDict = self.shuffle(shuffleDict=shuffleDict)        
        outDataSet = copy.deepcopy(self)
        for l in self.classIndexDict:
            classLen = len(self.classIndexDict[l])
            splitNum = int(classLen * scale)
            
            tmpIdx = np.random.randint(0,classLen,splitNum)
            
            outDataSet.classIndexDict[l] = np.array(outDataSet.classIndexDict[l])[tmpIdx]
        # self.regenerateQueryIndex()
        outDataSet.regenerateQueryIndex()
        
        return outDataSet,shuffleDict
    
    def generateSubDatasetNoRepeat(self, scale=0.2, shuffleDict=None):
        shuffleDict = self.shuffle(shuffleDict=shuffleDict)        
        outDataSet = copy.deepcopy(self)
        for l in self.classIndexDict:
            splitNum = int(len(self.classIndexDict[l]) * scale)
            # self.classIndexDict[l] = self.classIndexDict[l][splitNum:]
            outDataSet.classIndexDict[l] = outDataSet.classIndexDict[l][:splitNum]
        # self.regenerateQueryIndex()
        outDataSet.regenerateQueryIndex()
        
        return outDataSet,shuffleDict
    
    def splitDataOfSpcLabel(self,l, scale=0.2, shuffleDict=None):
        #This is a subfunction of DataSetWrapper
        #This function is used for balance the samples of different labels
        shuffleDict = self.shuffle(shuffleDict=shuffleDict)        
        splitNum = int(len(self.classIndexDict[l]) * scale)
        self.classIndexDict[l] = self.classIndexDict[l][splitNum:]
        self.regenerateQueryIndex()
        return 
    
    def float_to_oneHot(self,x):
        maxVal = max(torch.max(x),torch.max(self.feaOneHotBins)) * 2
        diff = x.view(-1,1) - self.feaOneHotBins #shape: numFea * numBins
        diff[diff<0] = maxVal
        oneHotArr = ((diff - diff.min(axis=1)[0].view(-1,1))==0).type(torch.float32)
        return oneHotArr #numFea * numBins
    
    def to_oneHot(self, labelName):
        X = self.labelDict[labelName]
        out = torch.zeros([self.classNum])
        out[X] += 1
        return out
    
    # def select_feature(self,feature_index):
    #     for i in range(len(self.dataList)):
    #         self.dataList[i] =  (self.dataList[i][0],self.dataList[i][1],self.dataList[i][2],self.dataList[i][3][feature_index],self.dataList[i][4][feature_index])
            
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.queryIndex)
    
    def __getitem__(self, index,extra_param=None):
        indexCount,fileName,label,arr,missMask = self.dataList[self.queryIndex[index]]
        #computing here to reduce memory occupancy
        # onehotArr = torch.tensor(self.feaGenerator.OnehotEncoding1D(fastaSeq),dtype=torch.float32)
        # return self.float_to_oneHot(arr), self.to_oneHot(label)
        if not index in self.labelOneHotDict:
            self.labelOneHotDict[index] = self.to_oneHot(label)
        label_onehot = self.labelOneHotDict[index]
        
        if extra_param is not None:
            arr=arr[extra_param]
            missMask=missMask[extra_param]
        
        return arr,missMask, label_onehot
    

class DataSetWrapper(torch.utils.data.Dataset):
    def __init__(self,select_feture=None):
        self.datasetList = []
        self.cutNumList = [0]
        self.currLabelSet = set()
        self.currLabelDict = {}
        self.numFeaBins = 16
        self.select_feture = select_feture
    
    def addSubDataSet(self, subDataset):
        self.datasetList.append(subDataset)
        if len(self.datasetList) > 1:
            self.cutNumList.append(self.datasetList[-2].__len__() + self.cutNumList[-1])
            
        #update class
        self.currLabelSet = self.currLabelSet | subDataset.labelSet
        labelNum = 0
        for l in self.currLabelSet:
            self.currLabelDict[l] = labelNum
            labelNum += 1
            
        # for tmpObj in self.datasetList:
        #     tmpObj.classNum = len(self.currLabelSet)
        #     tmpObj.labelDict = self.currLabelDict
        self.updateLabelDict(self.currLabelDict)
    
    def mergeOtherDataSetWrapper(self, otherDataSetWrapper):
        for subDataset in otherDataSetWrapper.datasetList:
            self.addSubDataSet(subDataset)
    
    
    
    
    def getLabelDict(self):
        return self.currLabelDict
    
    def updateLabelDict(self, dictIn):
        self.currLabelDict = dictIn
        for tmpObj in self.datasetList:
            tmpObj.classNum = len(self.currLabelSet)
            tmpObj.labelDict = dictIn
    
    def getFeaBins(self,feaBins=None):
        numBins = self.numFeaBins
        # feaNum = len(self.datasetList[0].dataList[-1])
        if feaBins is None:
            binMin = None
            binMax = None
            for subDataset in self.datasetList:
                for idx in subDataset.queryIndex:
                    # arr = torch.tensor(subDataset.dataList[idx],dtype=torch.float32)
                    arr = subDataset.dataList[idx][-1]
                    if binMin is None:
                        binMin = arr.clone()
                    if binMax is None:
                        binMax = arr.clone()
                    diff = arr < binMin
                    binMin[diff] = arr[diff]
                    diff = arr > binMax
                    binMax[diff] = arr[diff]
            stepSize = ((binMax - binMin) / numBins).view([-1,1])
            outArr = binMin.view([-1,1])
            oriArr = outArr.clone()
            for i in range(numBins-1):
                outArr = torch.cat([outArr,oriArr + stepSize * (i+1)],axis=1)
            feaBins = outArr
        
        
        for tmpObj in self.datasetList:
            tmpObj.feaOneHotBins = feaBins
        return feaBins
    
    def imputeMissValue(self,imputer=None):
        
        # xAll = []
        xSata = []
        maskAll = []
        yAll = []
        x,m,y = self.__getitem__(0)
        # totalNum_perFea = np.zeros(x.size(0))
        for i in range(self.__len__()):
            x,m,y = self.__getitem__(i)
            # tmpx = x.masked_fill(m,torch.nan)
            # xAll.append(tmpx)
            xSata.append(x)
            yAll.append(y)
            maskAll.append(m)
        # xAll = torch.stack(xAll,dim=0)
        xSata = torch.stack(xSata,dim=0)
        yAll = torch.stack(yAll,dim=0).argmax(dim=1)
        maskAll = torch.stack(maskAll,dim=0)
        
        # avg = xSata.sum(axis=0) / totalNum_perFea
        # std = (((xSata - avg) ** 2).sum(axis=0) / totalNum_perFea) ** 0.5
        # xAll = (xAll - avg.reshape([1,-1])) / std.reshape([1,-1])
        
        avg = xSata.sum(dim=0) / (~maskAll).sum(dim=0)
        x2_avg = (xSata - avg.view([1,-1])) ** 2
        x2_avg.masked_fill_(maskAll,0)
        std = (x2_avg.sum(dim=0) / (~maskAll).sum(dim=0)) ** 0.5
        xAll = (xSata - avg.view([1,-1])) / std.view([1,-1])
        xAll.masked_fill_(maskAll,torch.nan).numpy()
        
        if imputer is None:
            imputer = KNNImputer(n_neighbors=15, weights="uniform")
            imputer.fit(xAll)
        xFill = imputer.transform(xAll)
        for index in range(self.__len__()):
            if index < 0:
                index = index % self.__len__()
            diff = index - np.array(self.cutNumList)
            sdIndex = np.argsort(diff)[np.sort(diff)>=0][0]
            subItemIndex = diff[sdIndex]
            tmpDataSet = self.datasetList[sdIndex]
            oriData = list(tmpDataSet.dataList[tmpDataSet.queryIndex[subItemIndex]])
            oriData[3] = torch.tensor(xFill[index,:])
            # indexCount,fileName,label,arr,missMask = self.dataList[self.queryIndex[index]]
            tmpDataSet.dataList[tmpDataSet.queryIndex[subItemIndex]] = tuple(oriData)
        return imputer
    
    def balanceMissRate(self,missRate_temp = None):
        xSata = []
        maskAll = []
        yAll = []
        x,m,y = self.__getitem__(0)
        # totalNum_perFea = np.zeros(x.size(0))
        for i in range(self.__len__()):
            x,m,y = self.__getitem__(i)
            # tmpx = x.masked_fill(m,torch.nan)
            # xAll.append(tmpx)
            xSata.append(x)
            yAll.append(y)
            maskAll.append(m)
        # xAll = torch.stack(xAll,dim=0)
        xSata = torch.stack(xSata,dim=0)
        yAll = torch.stack(yAll,dim=0)
        maskAll = torch.stack(maskAll,dim=0)
        
        # classLabelAccumulate = yAll.sum(dim=0)
        missRate = torch.zeros(yAll.size(1),maskAll.size(1)) #numY * numFea
        for i in range(yAll.shape[1]):
            tmpY = yAll[:,i].clone().flatten().bool()
            tmpMask = maskAll[tmpY,:]
            tmpMissRate = tmpMask.sum(dim=0) / tmpMask.size(0)
            missRate[i,:] += tmpMissRate
        
        if missRate_temp is None:
            missRate_temp = missRate.max(dim=0).values #numFea
        
        tempMask = maskAll.clone().detach()
        for j in range(yAll.size(1)):
            #the loop of classes
            #every class is regard as the positive at its loop
            missRate_diff = missRate[j,:] - missRate_temp     #numFea       
            for i in range(missRate_diff.size(0)):
                missRate_i = missRate_diff[i]
                if missRate_i < 0:
                    tmpY = yAll[:,j].clone().flatten().bool()
                    numMasks = (-1 * missRate_i * tmpY.float().sum()).round().int()
                    if numMasks > 0:
                        newArr = torch.zeros(maskAll.size(0)).int()
                        newArr.masked_fill_(~tmpY.flatten(),1)
                        newArr.masked_fill_(maskAll[:,i].flatten(),1)
                        addMask = self.fillMask(newArr,numMasks)
                        tempMask[:,i] = tempMask[:,i] | addMask
                
        maskAll = tempMask.clone()
        xSata.masked_fill_(maskAll,0)
        for index in range(self.__len__()):
            if index < 0:
                index = index % self.__len__()
            diff = index - np.array(self.cutNumList)
            sdIndex = np.argsort(diff)[np.sort(diff)>=0][0]
            subItemIndex = diff[sdIndex]
            tmpDataSet = self.datasetList[sdIndex]
            oriData = list(tmpDataSet.dataList[tmpDataSet.queryIndex[subItemIndex]])
            oriData[4] = maskAll[index,:].detach().clone()
            oriData[3] = xSata[index,:].detach().clone()
            # indexCount,fileName,label,arr,missMask = self.dataList[self.queryIndex[index]]
            tmpDataSet.dataList[tmpDataSet.queryIndex[subItemIndex]] = tuple(oriData)
            
        return missRate_temp
    

    def balanceSampleFromLabel(self):
        xSata = []
        maskAll = []
        yAll = []
        x,m,y = self.__getitem__(0)
        # totalNum_perFea = np.zeros(x.size(0))
        for i in range(self.__len__()):
            x,m,y = self.__getitem__(i)
            # tmpx = x.masked_fill(m,torch.nan)
            # xAll.append(tmpx)
            xSata.append(x)
            yAll.append(y)
            maskAll.append(m)
        # xAll = torch.stack(xAll,dim=0)
        xSata = torch.stack(xSata,dim=0)
        yAll = torch.stack(yAll,dim=0)
        maskAll = torch.stack(maskAll,dim=0)
        
        samplesFromLabel = yAll.sum(dim=0)
        minimalClassSampleNum = samplesFromLabel.min()
        balanceRate = 1 - (minimalClassSampleNum / samplesFromLabel)
        
        numLabelDict = {}
        for k,v in self.currLabelDict.items():
            numLabelDict[v] = k
        
        for labelNum, reduceRate in enumerate(balanceRate):
            if reduceRate > 1e-6:
                l = numLabelDict[labelNum]
                for tmpDataSet in self.datasetList:
                    tmpDataSet.splitDataOfSpcLabel(l, scale=reduceRate, shuffleDict=None)
        return
        
    def fillMask(self,maskMat,numMasks):
        newMask = torch.rand_like(maskMat.float())
        newMask.masked_fill_(maskMat.bool(),0)
        try:
            filterThres = newMask.flatten().topk(numMasks).values[-1]
        except:
            assert False
        newMask = newMask >= filterThres
        return newMask 
    
    def __len__(self):
        out = 0
        for subDataset in self.datasetList:
            out += subDataset.__len__()
        return out
    
    def __getitem__(self, index):
        # if self.datasetList[0].feaOneHotBins is None:
        #     self.getFeaBins()
        # extra_param=self.select_feture
        # print('extra_param:',extra_param)
        if index < 0:
            index = index % self.__len__()
        diff = index - np.array(self.cutNumList)
        sdIndex = np.argsort(diff)[np.sort(diff)>=0][0]
        subItemIndex = diff[sdIndex]
        return self.datasetList[sdIndex].__getitem__(subItemIndex)

def traverse_folders(root_folder):
    csv_files = []  # 存储.csv文件的列表

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    return csv_files

def getAllDataSet(dataGenerator,fillNan=True,returnOneHotY=False):
    xOut = []
    yOut = []
    maskOut = []
    for x,m,y in dataGenerator:
        if fillNan:
            tmpx = x.masked_fill(m,torch.nan)
        else:
            tmpx = x
        xOut.append(tmpx)
        maskOut.append(m)
        yOut.append(y)
    xOut = torch.cat(xOut,dim=0)
    if returnOneHotY:
        yOut = torch.cat(yOut,dim=0)
    else:
        yOut = torch.cat(yOut,dim=0).argmax(dim=1)
    maskOut = torch.cat(maskOut,dim=0)
    xOut.masked_fill_(maskOut,0)
    return xOut, yOut,maskOut
  