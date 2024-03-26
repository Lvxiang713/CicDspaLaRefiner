# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:49:46 2023
@author: LvXiang, JingRunyu
"""
import numpy as np
import torch
import re, copy, os, time,shutil,json
from torch import nn
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score,confusion_matrix,matthews_corrcoef 
from sklearn import metrics
from scipy import stats
import pickle
import pandas as pd
import gzip




class MyLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.2, smooth: float = 1e-5, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction
    def forward(self, logits, targets):
        #logits(p): n*c
        #targets(l): n*c
        #here c = 2 !
        w = (1.0 / targets.sum(dim=0,keepdim=True) * targets.sum()) ** self.alpha
        w = w / w.sum()
        loss = -1 * w * torch.functional.F.log_softmax(logits,dim=-1) * targets
        loss = loss.sum(dim=-1)
        # loss = loss.exp()
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()

class WDD(nn.Module):
    '''
    This module serves as the core component of DSpaLaRefiner, 
    designed to optimize Landmarks and define the distance metrics between data and Landmarks. 
    It's important to note that each category should correspond to a WDD (Weighted Distance Descriptor) object,
    meaning that the total number of initialized WDDs should match the number of classification categories. 
    The core parameters of this model include:
    w: Represents the feature weights for each Landmark. (nt*f)
    t: Denotes the positions of the Landmarks. (nt*f)
    gamma: Refers to the gamma parameter in radial basis distance computation.
    '''
    def __init__(self,input_features,alpha,gamma,delta,device = 'cuda:0',
                 dtype=torch.float32, earlyStopThres = 100,computeMode='Di',
                 numT = 4, dropoutRate=0.15):##input_fetures is the dimension of features
        super().__init__()
        self.miss_head = 512
        self.miss_FC = nn.Linear(self.miss_head,1,bias=False)
        torch.nn.init.zeros_(self.miss_FC.weight)
        self.input_features = input_features
        self.w = nn.Parameter(2*torch.rand([numT,input_features], dtype=dtype,device=device) - 1, requires_grad=True)
        self.w_miss_pu = nn.Parameter(2*torch.rand([numT,input_features], dtype=dtype,device=device) - 1, requires_grad=True)
        self.missW = nn.Parameter(2*torch.rand([1,input_features], dtype=dtype,device=device) - 1, requires_grad=True)
        self.random_features = nn.Parameter(2*torch.rand([1,input_features], dtype=dtype,device=device) - 1, requires_grad=True)
        self.t = None
        self.numT = numT
        self.loss = None
        self.relu =  nn.ReLU()
        self.alpha = nn.Parameter(torch.tensor(alpha,dtype=dtype).to(device),requires_grad=True)        
        self.device = device
        self.infFillNum = 1e5
        self.divide_bias=1e-30
        self.computeMode = computeMode
        self.dtype = dtype
        self.initialized = False
        self.dropoutRate = dropoutRate
        if dropoutRate:
            self.dropout = nn.Dropout(dropoutRate)
        self.register_buffer('data_avg',torch.zeros(input_features))
        self.register_buffer('data_std',torch.zeros(input_features))
        self.register_buffer('sum_byf',torch.zeros([numT,input_features],dtype=dtype))
        self.register_buffer('sum_mask',torch.zeros([input_features],dtype=dtype))
        self.register_buffer('data_avgLabel',torch.zeros(input_features))
        self.register_buffer('data_stdLabel',torch.zeros(input_features))
        self.generateT()
        
    def freezingPara(self):
        self.t.requires_grad_(False)
        self.w.requires_grad_(False)
        
    def thawPara(self):
        self.t.requires_grad_(True)
        self.w.requires_grad_(True)
        
    def generateT(self):
        # self.t = nn.Parameter(6*torch.rand([self.numT,self.input_features], dtype=self.dtype,device=self.device) - 3, requires_grad=True)
        random_arrays = [torch.normal(mean=self.data_avgLabel[i], std=self.data_stdLabel[i], size=(self.numT,),dtype=self.dtype,device=self.device) for i in range(len(self.data_avgLabel))]   
        self.t = nn.Parameter(torch.stack(random_arrays, dim=1), requires_grad=True)
        # print(self.t)
        
        
    def sampleInitPos(self,dataGenerator):
        if self.initialized:
            return
        toReinit = True
        retryNum = 0
        with torch.no_grad():
            while toReinit:
                retryNum += 1
                toReinit = False
                for xb, missMask, yb in dataGenerator:
                    xb, missMask, yb = xb.type(self.dtype).to(self.device),missMask.to(self.device), yb.type(self.dtype).to(self.device)
                    xb = self.norm(xb)
                    xb.masked_fill_(missMask,0)
                    tmpLoss = self.cmp(xb,missMask,yb,filterNan=False) #set filterNan to False
                    if torch.isnan(tmpLoss) :
                        print(tmpLoss)
                        toReinit = True
                        self.generateT()
                        break
                assert retryNum < 1000
        self.initialized = True
        
    def distDi(self,xb,mask=None,w=None):
        #xb:n * f 
        #t:nt * f
        if not w is None:
            tmpW = processW(w).unsqueeze(1) #nt * 1 * f
        else:
            tmpW = processW(self.w).unsqueeze(1) #nt * 1 * f
        
        diff = self.t.unsqueeze(1) - xb.unsqueeze(0) # nt * n * f
        if not mask is None:
            diff = diff.masked_fill(mask.unsqueeze(0),0)
        tmp_diff = (tmpW * diff ) ** 2 #nt * n* f
        notmask_sum_byf =  torch.logical_not(mask).sum(dim=0)
        tmp_diff_sum = tmp_diff.sum(1)
        tmp_diff_mean_byf = tmp_diff_sum/(notmask_sum_byf+self.divide_bias)
        self.sum_byf += tmp_diff_sum.detach()
        self.sum_mask += notmask_sum_byf.detach()
        return tmp_diff.sum(axis=-1),tmp_diff_mean_byf,notmask_sum_byf,tmp_diff # nt * n,nt * f, f
    
    def distBf(self,xb,mask=None):
        tmpW = processW(self.w)
        return tmpW * torch.abs(self.t - xb)  
    
    def forward(self,xb,missMask,yb,filterNan=True,posPosition=0):
        return self.cmp(xb,missMask,yb,filterNan=filterNan,posPosition=posPosition)
    
    def cmp(self,xb,missMask,yb,filterNan=True,posPosition=0):
        if self.computeMode == 'Di':
            return self.cmpDi(xb,missMask,yb,filterNan=filterNan,posPosition=posPosition)
        elif self.computeMode == 'Bf':
            return self.cmpBf(xb,missMask,yb,filterNan=filterNan)
        else:
            return self.cmpNormal(xb,missMask,yb,filterNan=filterNan)
    
    def norm(self,xb):
        return (xb - self.data_avg) / self.data_std
    
    
    def cmpNormal(self,xb,missMask,yb,filterNan=True):
        dist,mean_byf,mask_byf = self.distDi(xb)
        #limit the too large distances
        if filterNan:
            nanMask = torch.isnan(dist)
            dist = dist.masked_fill(nanMask,self.infFillNum)
            nanMask = torch.isinf(dist)
            dist = dist.masked_fill(nanMask,self.infFillNum)
            nanMask = dist > self.infFillNum
            dist = dist.masked_fill(nanMask,self.infFillNum)
        Pr_t_xb_ij = 1 - torch.exp(-1*self.gamma*dist )
        if filterNan:
            nanMask = torch.isnan(Pr_t_xb_ij)
            dist = dist.masked_fill(nanMask,1e-5)
            nanMask = torch.isinf(Pr_t_xb_ij)
            dist = dist.masked_fill(nanMask,1e-5)
        posMask = yb[:,1].clone().flatten() #the second col is the positive
        posSign = posMask*-2 + 1 #neg -> 1, pos -> -1
        Pr_t_xb = Pr_t_xb_ij * posSign + posMask
        out = torch.log(Pr_t_xb+1e-5).sum()
        return out
    
    def cmpBf(self,xb,missMask,yb,filterNan=True):
        dist_ij = self.distBf(xb)
        dist = torch.exp(-1 * self.gamma*dist_ij) ** self.delta
        dist = dist.masked_fill(missMask,1e-5)
     
        Pr_t_xb_ij = dist.sum(dim=1)
        posMask = yb[:,1].clone().flatten() #the second col is the positive
        posSign = posMask*-2 + 1 #neg -> 1, pos -> -1
        Pr_t_xb = torch.exp(-1 * Pr_t_xb_ij) * posSign + posMask
        if filterNan:
            nanMask = torch.isnan(Pr_t_xb)
            Pr_t_xb = Pr_t_xb.masked_fill(nanMask,1e-5)
            nanMask = torch.isinf(Pr_t_xb)
            Pr_t_xb = Pr_t_xb.masked_fill(nanMask,1e-5)
        out = torch.log(Pr_t_xb+1e-5).sum()
        return out
    
    def cmpDi(self,xb,missMask,yb,filterNan=True,posPosition=0):
        #xb shape: n * f 
        #yb shape: n * 2
        if self.training:
            dist,mean_byf,mask_byf,_ = self.distDi(xb,mask=missMask) #nt * n 
        else:
            dist,_,_,_ = self.distDi(xb,mask=missMask) 
            mean_byf = self.sum_byf / self.sum_mask.unsqueeze(0)
        # print('debug_dist:',dist)
        
        #limit the too large distances
        if filterNan:
            nanMask = torch.isnan(dist)
            dist = dist.masked_fill(nanMask,self.infFillNum)
            nanMask = torch.isinf(dist)
            dist = dist.masked_fill(nanMask,self.infFillNum)
            nanMask = dist > self.infFillNum
            dist = dist.masked_fill(nanMask,self.infFillNum)
        
        missMask_unsq = missMask.unsqueeze(0).repeat(self.w.shape[0],1,1)#nt*n*f
        dist_punish  = mean_byf.unsqueeze(1).repeat(1,dist.shape[1],1).masked_fill(~missMask_unsq,0).sum(-1) #nt*n
        Pr_t_xb_ij =  1 - torch.exp(-1*torch.abs(self.gamma)*(dist+dist_punish))
        
        if filterNan:
            nanMask = torch.isnan(Pr_t_xb_ij)
            dist = dist.masked_fill(nanMask,1e-5)
            nanMask = torch.isinf(Pr_t_xb_ij)
            dist = dist.masked_fill(nanMask,1e-5)
            
        posMask = yb[:,posPosition].clone().flatten().view([1,-1]) #1 * n, the second col is the positive
        posSign = posMask*-2 + 1 #1 * n, neg -> 1, pos -> -1
        Pr_t_xb = Pr_t_xb_ij * posSign + posMask

        out = torch.log(Pr_t_xb+1e-5).sum()
        
        if torch.isnan(out):
            global Pr_t_xb_debug,Pr_t_xb_ij_debug,dist_debug
            Pr_t_xb_debug =  Pr_t_xb 
            Pr_t_xb_ij_debug = Pr_t_xb_ij
            dist_debug = dist
            assert False
        # print(out)
        return out / self.numT
        
class AbsLinear(nn.Module):
    def __init__(self, in_features, out_features,bias = True):
        super(AbsLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bias = bias
        torch.nn.init.normal_(self.linear.weight,mean=0,std=1)
    def forward(self, x):
        weights = torch.abs(self.linear.weight)
        weighted_sum_abs = torch.matmul(x, weights.T)
        if self.bias == True:
            result = torch.sum(weighted_sum_abs, dim=1, keepdim=True) + self.linear.bias
        else:
            result = torch.sum(weighted_sum_abs, dim=1, keepdim=True) 
        return result    

class WrappedModel(nn.Module):
    '''This module defines the structure of the forward procession, 
    primarily comprising the WDD (Weighted Distance Descriptor) module and the VotingLayer module, 
    along with the activation function for the probability output layer. 
    The number of instances of both modules is equivalent to the number of classes or labels, denoted as numClass.
    
    The VotingLayer module, in particular, essentially functions as a fully connected layer, 
    where the weights correspond to the significance of Landmarks.
    In other words, the absolute magnitude of the weights indicates the extent to which a particular Landmark influences the final classification outcome.'''
    
    def __init__(self,input_features,numClass,gamma=2**0,alpha=2**0,delta=2**0,device = 'cuda:0',
                 dtype=torch.float32, earlyStopThres = 100,computeMode='Di',
                 numT=32,batchSize=128,dropoutRate=0.15,distScale=None,miniBatch=False,
                 useScheduler=False,setGlobalW=False,lastAct=nn.Softmax(-1),
                 recordEpochTargets=False,performWDDOpt=True,recordEpochStepSize=10,
                 distLossTopK=None,outSavePath='out',Tmp_mean_byf=None,WDDlossRate=0.1,recordbestEpoch=False,record_thrMCC=None):
        ##input_fetures is the dimension of features
        super().__init__()
        self.outSavePath = outSavePath
        self.input_features = input_features
        self.numClass = numClass
        self.WDDlossRate = WDDlossRate
        assert isinstance(numT, int) or isinstance(numT, list)
        if isinstance(numT, int):
            self.numT = [numT] * self.numClass
        else:
            assert len(numT) == self.numClass
            
        self.WDDList = nn.ModuleList([])
        for i in range(self.numClass):
            self.WDDList.append(WDD(input_features,alpha,gamma,delta,device = device,
                         dtype=dtype, earlyStopThres = earlyStopThres,
                         numT=numT[i], dropoutRate=dropoutRate, computeMode=computeMode))
        
        global_gamma = nn.Parameter(torch.tensor(gamma,dtype=dtype).to(device),requires_grad=True) 
        global_delta = nn.Parameter(torch.tensor(delta,dtype=dtype).to(device),requires_grad=True)
        global_alpha =  nn.Parameter(torch.tensor(alpha,dtype=dtype).to(device),requires_grad=True)
        for WDDObj in self.WDDList:
            WDDObj.gamma = global_gamma
            WDDObj.delta = global_delta
            WDDObj.alpha = global_alpha
        if setGlobalW:
            self.globalW = nn.Parameter((torch.full((1,input_features,), 1/input_features)).type(dtype).to(device), requires_grad=True)
            self.globalMissW = nn.Parameter((torch.full((1,input_features,), 1/input_features)).type(dtype).to(device), requires_grad=True)
            for WDDObj in self.WDDList:
                WDDObj.w = self.globalW
                WDDObj.missW = self.globalMissW
 
        self.earlyStopThres = earlyStopThres
        self.computeMode = computeMode

        self.bestValLoss = None
        # self.earlyStopThres = 200

        
        self.record_thrMCC = record_thrMCC

        
        self.bestEpoch = 0
        
        self.bestValLoss_T = None
        self.bestEpoch_T = 0
        self.best_t = None
        self.best_w = None
        
        self.dropoutRate = dropoutRate
        if dropoutRate:
            self.dropout = nn.Dropout(self.dropoutRate)
        
        self.numT = numT
        self.discLayers = nn.ModuleList([])
        for i in range(self.numClass):
            self.discLayers.append(AbsLinear(numT[i],1,bias=False))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.lastAct = lastAct
        self.device = device
        self.dtype = dtype
        self.relu = nn.ReLU()
        self.lossFunc = nn.CrossEntropyLoss(reduction = 'sum')
        self.lossFunc_aux = MyLoss(reduction = 'sum')
        # self.lossFunc = dice
        self.vote_scale = 1.
        #terms for dist loss
        self.distLossP = 2
        self.distScale = distScale
        self.totalTNum = np.sum(self.numT)
        self.miniDistEps = 1e-10
        self.distLossTopK = distLossTopK
        self.batchSize = batchSize #only for distLoss
        self.miniBatch = miniBatch
        self.useScheduler = useScheduler
        self.performWDDOpt = performWDDOpt
        self.recordEpochTargets = recordEpochTargets
        self.recordbestEpoch = recordbestEpoch
        self.recordEpochStepSize = recordEpochStepSize
        self.register_buffer('data_avg',torch.zeros(input_features))
        self.register_buffer('data_std',torch.zeros(input_features))
        self.register_buffer('data_avgAllLabel',torch.zeros([numClass,input_features]))
        self.register_buffer('data_stdAllLabel',torch.zeros([numClass,input_features]))

            
    def norm(self,xb):
        return (xb - self.data_avg) / self.data_std
    
    def processStatistic(self,trainGenerator):
        tmpDataList = []
        tmpMaskList = []
        tmpLabelList = []
        
        for xb,missMask, yb in trainGenerator:
            tmpDataList.append(xb)
            tmpMaskList.append(missMask)
            tmpLabelList.append(yb)
        tmpArr = torch.cat(tmpDataList,axis=0)
        if self.computeMode == 'Normal':
            avg = tmpArr.sum(axis=0) / tmpArr.size(0)
            diff = (tmpArr - avg)**2
            std = (diff.sum(axis=0) / tmpArr.size(0)) ** 0.5
        else:
            tmpMask = torch.cat(tmpMaskList,axis=0)
            tmpLabel = torch.cat(tmpLabelList,axis=0)
            avg = tmpArr.sum(axis=0) / (tmpArr.size(0) - tmpMask.sum(axis=0))
            diff = (tmpArr - avg)**2
            diff.masked_fill_(tmpMask,0)
            std = (diff.sum(axis=0) / (tmpArr.size(0) - tmpMask.sum(axis=0)))**0.5
            
            avgByLabel = [tmpArr[tmpLabel[:,i]==1].sum(axis=0) / (tmpArr[tmpLabel[:,i]==1].size(0) - tmpMask[tmpLabel[:,i]==1].sum(axis=0)) for i in range(yb.shape[-1])]
            diffByLabel = [(tmpArr[tmpLabel[:,i]==1] - avgByLabel[i])**2 for i in range(yb.shape[-1])]
            diffByLabel = [diffByLabel[i].masked_fill_(tmpMask[tmpLabel[:,i]==1],0) for i in range(yb.shape[-1])]
            stdBylabel = [(diffByLabel[i].sum(axis=0) / (tmpArr[tmpLabel[:,i]==1].size(0) - tmpMask[tmpLabel[:,i]==1].sum(axis=0)))**0.5 for i in range(yb.shape[-1])]
            
        avg = avg.type(self.dtype).to(self.device)
        std = std.type(self.dtype).to(self.device)
        avgByLabeltensor = torch.concat([i.unsqueeze(0) for i in avgByLabel],axis=0).type(self.dtype).to(self.device)
        stdByLabeltensor = torch.concat([i.unsqueeze(0) for i in stdBylabel],axis=0).type(self.dtype).to(self.device)
        

        
        self.register_buffer('data_avg',avg)
        self.register_buffer('data_std',std)
        
        for WDDObj in self.WDDList:
            WDDObj.register_buffer('data_avg',avg)
            WDDObj.register_buffer('data_std',std)
        
        avgByLabeltensorNorm = self.norm(avgByLabeltensor)
        stdByLabeltensorNorm = stdByLabeltensor/std
        
        self.register_buffer('data_avgAllLabel',avgByLabeltensorNorm)
        self.register_buffer('data_stdAllLabel',stdByLabeltensorNorm)
        
        for idx,WDDObj in enumerate(self.WDDList):
            WDDObj.register_buffer('data_avgLabel',avgByLabeltensorNorm[idx,:])
            WDDObj.register_buffer('data_stdLabel',stdByLabeltensorNorm[idx,:])
            WDDObj.generateT()
        
    def saveTargets(self,epochNum):
        savePath = self.outSavePath + os.sep + 'targetHistory'
        os.makedirs(savePath,exist_ok=True)
        saveList = []
        for i in range(self.numClass):
            targetPos = self.WDDList[i].t.clone().detach().cpu()
            targetW = self.WDDList[i].w.clone().detach().cpu()
            targetDiscW = self.discLayers[i].linear.weight.clone().detach().cpu()
            saveList.append([targetPos,targetW,targetDiscW])
        
        fileName = '%s/epoch%d.pt' %(savePath,epochNum)
        with open(fileName,'wb') as FIDO:
            pickle.dump(saveList,FIDO)
    
    def computeLoss(self,xPred,y,smoothNum=0.05):
        y = y + smoothNum
        y = y / y.sum(dim=-1).view([-1,1])
        return self.lossFunc(xPred,y)

    def computeDistLoss(self):
        TList = []
        for tmpObj in self.WDDList:
            TList.append(tmpObj.t)
        distLoss = None
        TPoints = torch.cat(TList,dim=0) #totalNt * nf
        for i in range(int(np.ceil(TPoints.size(0)/self.batchSize))):
            sPos = i * self.batchSize
            ePos = (i+1) * self.batchSize
            tmpPoints = TPoints[sPos:ePos,:].clone()
            
            squaredDistMat = ((torch.abs(tmpPoints.unsqueeze(1) - TPoints.unsqueeze(0))**self.distLossP) * self.distScale).sum(dim=-1)
            tmpEyeMask = torch.zeros_like(squaredDistMat,dtype=bool,device=self.device)
            tmpEyeMask[torch.arange(tmpPoints.size(0)),torch.arange(sPos,sPos+tmpPoints.size(0))] = 1
            squaredDistMat = squaredDistMat.masked_fill(tmpEyeMask,1e10)
            if not self.distLossTopK is None:
                topKThres = torch.topk(squaredDistMat.detach(),self.distLossTopK,dim=1,largest=False).values[:,-1].view([squaredDistMat.size(0),1])
                squaredDistMat = squaredDistMat.masked_fill(squaredDistMat>topKThres,1e10)
            
            zeroMask = squaredDistMat < self.miniDistEps
            squaredDistMat = squaredDistMat.masked_fill(zeroMask,self.miniDistEps)
            invDistMat = squaredDistMat ** -1
            currLoss = invDistMat.sum()
            currLoss.backward()
            if distLoss is None:
                distLoss = currLoss.item()
            else:
                distLoss = distLoss + currLoss.item()
            
                
        if self.distLossTopK:
            return distLoss / self.totalTNum / self.distLossTopK
        return distLoss / self.totalTNum ** 2
        
        
    
    def generateMask(self,missMask,yb):
        y_pos = yb[:,1].clone().flatten().bool()
        y_neg = ~y_pos
        mask_pos = missMask[y_pos,:]
        mask_neg = missMask[y_neg,:]
        missRate_pos = mask_pos.sum(dim=0)/mask_pos.size(0)
        missRate_neg = mask_neg.sum(dim=0)/mask_neg.size(0)
        #missRate_diff > 0 : add negMask
        #missRate_diff < 0 : add posMask
        missRate_diff = missRate_pos - missRate_neg
        
        tempMask = missMask.clone().detach()
        for i in range(missRate_diff.size(0)):
            missRate_i = missRate_diff[i]
            if missRate_i > 0:
                #add negMask
                numMasks = (missRate_i * y_neg.float().sum()).round().int()
                if numMasks > 0:
                    newArr = torch.zeros(missMask.size(0)).int().to(self.device)
                    newArr.masked_fill_(y_pos.flatten(),1)
                    newArr.masked_fill_(missMask[:,i].flatten(),1)
                    addMask = self.fillMask(newArr,numMasks)
                    tempMask[:,i] = tempMask[:,i] | addMask
            else:
                #add posMask
                numMasks = (-1 * missRate_i * y_pos.float().sum()).round().int()
                if numMasks > 0:
                    newArr = torch.zeros(missMask.size(0)).int().to(self.device)
                    newArr.masked_fill_(y_neg.flatten(),1)
                    newArr.masked_fill_(missMask[:,i].flatten(),1)
                    addMask = self.fillMask(newArr,numMasks)
                    tempMask[:,i] = tempMask[:,i] | addMask
                
        return tempMask
        
        
    def generateMaskOld(self,missMask,yb):
        y_pos = yb[:,1].clone().flatten().bool()
        y_neg = ~y_pos
        mask_pos = missMask[y_pos,:]
        mask_neg = missMask[y_neg,:]
        missRate_pos = mask_pos.sum()/mask_pos.size(0)/mask_pos.size(1)
        missRate_neg = mask_neg.sum()/mask_neg.size(0)/mask_neg.size(1)
        missRate_rowwise = missMask.float().sum(dim=1) / missMask.size(1)
        
        if missRate_pos > missRate_neg:
            numMasks = (mask_neg.size(0) * mask_neg.size(1) * (missRate_pos - missRate_neg)).round().int()
            if numMasks > 0:
                tempMask = missMask.clone().detach()
                tempMask[y_pos,:] = 1
                tempMask[missRate_rowwise > missRate_pos,:] = 1 
                addMask = self.fillMask(tempMask,numMasks)
                missMask = missMask | addMask
        else:
            numMasks = (mask_pos.size(0) * mask_pos.size(1) * (missRate_neg - missRate_pos)).round().int()
            if numMasks > 0:
                tempMask = missMask.clone().detach()
                tempMask[y_neg,:] = 1
                tempMask[missRate_rowwise > missRate_neg,:] = 1 
                addMask = self.fillMask(tempMask,numMasks)
                missMask = missMask | addMask
        return missMask
        
    def fillMask(self,maskMat,numMasks):
        newMask = torch.rand_like(maskMat.float())
        newMask.masked_fill_(maskMat.bool(),0)
        try:
            filterThres = newMask.flatten().topk(numMasks).values[-1]
        except:
            assert False
        newMask = newMask >= filterThres
        return newMask 
    
    def maskWithRand(self,x,mask,scale=1e-5):
        return x.masked_fill(mask,0) + (2 * torch.rand_like(mask.float(), dtype=self.dtype, device=self.device) - 1).masked_fill(~mask,0) * scale
    
    def collect_and_save_data(self, e, valMcc, trainGenerator, valGenerator):
        self.saveTargets(e)
        savePath = self.outSavePath + os.sep + 'dataSetCollect_epoch'
        os.makedirs(savePath, exist_ok=True)
        trainSetCollect = self.predict(trainGenerator, returnDataSet=True)[-1]
        valSetCollect = self.predict(valGenerator, returnDataSet=True)[-1]
        with gzip.open(self.savePathData + os.sep + 'epoch%d_MCC%.4f.pt' % (e, valMcc), 'wb') as FIDO:
            pickle.dump({'tr': trainSetCollect, 'val': valSetCollect}, FIDO)

    
    
    def processOneBatch(self,mod,xb,yb,missMask,TLoss,ClsLoss=None,labels=None,returnPredProb=False):
        if mod == 'T':
            xb = self.norm(xb)
            if not self.computeMode == 'Normal':
                xb = self.maskWithRand(xb, missMask,scale=0)
            batchTLoss =  0
            for i,WDDObj in enumerate(self.WDDList):
                batchTLoss = batchTLoss - WDDObj(xb,missMask,yb,posPosition=i)
            if self.training:
                batchTLoss.backward()
            if TLoss is None:
                TLoss = batchTLoss.item()
            else:
                #the loss of t should be producted accroding to the formula
                TLoss = TLoss + batchTLoss.item()
            return TLoss
        else:
            oriLabels,predLabels = labels
            xb = self.norm(xb)
            if not self.computeMode == 'Normal':
                xb = self.maskWithRand(xb, missMask,scale=0)

            batchTLoss =  0
            if self.performWDDOpt:
                for i,WDDObj in enumerate(self.WDDList):
                    batchTLoss = batchTLoss - WDDObj(xb,missMask,yb,posPosition=i)
         
            if self.computeMode == 'Bf':
                dist = self.WDDLayer.distBf(xb,missMask)
                dist = torch.exp(-1 * self.WDDLayer.gamma*dist) ** self.WDDLayer.delta
                dist = dist.masked_fill(missMask,0)
                dist = 1 - torch.exp(-1 * dist.sum(dim=1))
            else:
                probs = []
                if self.training:
                    tmp_mask_byf_list=[]
                    tmp_sum_byf_list = []
                if returnPredProb:
                    distProbs = []
                    distFProbs = []
                for i in range(self.numClass):
                    WDDObj = self.WDDList[i]
                    DiscLayer = self.discLayers[i]
                    missMask_unsq = missMask.unsqueeze(0).repeat(WDDObj.w.shape[0],1,1)
                    if self.training:
                        tmpDist,tmp_mean_byf,tmp_mask_byf,_ = WDDObj.distDi(xb,missMask)
                        # print(tmp_mean_byf.shape)
                        tmp_sum_byf = tmp_mean_byf*tmp_mask_byf 
                        tmp_sum_byf_list.append(tmp_sum_byf)
                        tmp_mask_byf_list.append(tmp_mask_byf)
                        tmp_dist_punish  = tmp_mean_byf.unsqueeze(1).repeat(1,tmpDist.shape[1],1).masked_fill(~missMask_unsq,0).sum(-1) #nt*n

                    else:                        
                        tmpDist,_,_,dist_f = WDDObj.distDi(xb,missMask)                        
                        tmp_dist_punish  = (WDDObj.sum_byf/WDDObj.sum_mask.unsqueeze(0)).unsqueeze(1).repeat(1,tmpDist.shape[1],1).masked_fill(~missMask_unsq,0).sum(-1) #nt*n

                    tmpDist =  torch.exp(-1*torch.abs(WDDObj.gamma)*(tmpDist+tmp_dist_punish))
                    if self.dropoutRate:
                        tmpDist = self.dropout(tmpDist)
                    
                    if returnPredProb:
                        distProbs.append(tmpDist.detach().cpu())
                        distFProbs.append(dist_f.detach().cpu())   
                    tmpProb = DiscLayer(tmpDist.transpose(1,0)) / WDDObj.numT * self.vote_scale
                    # print(DiscLayer.linear.weight)
                    probs.append(tmpProb)
                    # print(tmpProb)
            clsLabel = torch.cat(probs,dim=1) # n * 2
            #clsLabel = self.lastAct(clsLabel)
            batchClsLoss = self.computeLoss(clsLabel, yb)
            tmpLoss = self.WDDlossRate*batchTLoss + batchClsLoss 
            if self.training:
                tmpLoss.backward()

            #update loss
            if self.performWDDOpt:
                if TLoss is None:
                    TLoss = self.WDDlossRate*batchTLoss.item()
                else:
                    TLoss = TLoss + self.WDDlossRate*batchTLoss.item()
            else:
                TLoss = 0
            if ClsLoss is None:
                ClsLoss = batchClsLoss.item()
            else:
                ClsLoss = ClsLoss + batchClsLoss.item()
                
            oriLabels += list(torch.argmax(yb.detach().cpu(),dim=1).numpy())
            predLabels += list(torch.argmax(clsLabel.detach().cpu(),dim=1).numpy())
            if returnPredProb:
                return TLoss, ClsLoss,  clsLabel.detach().cpu(), distProbs,distFProbs
            return TLoss, ClsLoss
    
    
    def fitT(self,trainGenerator,valGenerator,epoches,lr,weight_decay=None,optimizerObj=None):
        if optimizerObj is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0) 
        else:
            optimizer = optimizerObj(self.parameters(), lr=lr, weight_decay=0) 
        for WDDObj in self.WDDList:
            WDDObj.sampleInitPos(trainGenerator)
        for e in range(epoches):
            self.train()
            optimizer.zero_grad()
            TLoss = None
            dataLen = 0
            for xb,missMask, yb in trainGenerator:
                xb,missMask, yb = xb.type(self.dtype).to(self.device),missMask.to(self.device), yb.type(self.dtype).to(self.device)
                TLoss = self.processOneBatch('T',xb,yb,missMask,TLoss,None,None)
                dataLen += xb.size(0)
                optimizer.step()       
            val_loss = self.valT(valGenerator)
            if self.bestValLoss_T is None:
                self.bestValLoss_T = val_loss
                self.bestEpoch_T = e
                self.best_t = [copy.deepcopy(WDDObj.t.cpu().clone().detach()) for WDDObj in self.WDDList]
                self.best_w = [copy.deepcopy(WDDObj.w.cpu().clone().detach()) for WDDObj in self.WDDList]
            else:
                if val_loss < self.bestValLoss_T:
                    self.bestValLoss_T = val_loss
                    self.bestEpoch_T = e
                    self.best_t = [copy.deepcopy(WDDObj.t.cpu().clone().detach()) for WDDObj in self.WDDList]
                    self.best_w = [copy.deepcopy(WDDObj.w.cpu().clone().detach()) for WDDObj in self.WDDList]
            print('********optimize T epoch:%d, lr:%2.e*******' %(e,lr))
            print(time.ctime())
            print('Training Set: %.4f' %(TLoss/ dataLen))
            print('Validation Set:',val_loss)
            print('Curr Best:',self.bestEpoch_T,self.bestValLoss_T)
            if e - self.bestEpoch_T > self.earlyStopThres:
                print('Early stopped.')
                break
        for i,WDDObj in enumerate(self.WDDList):
            WDDObj.t = nn.Parameter(self.best_t[i].to(self.device))
            WDDObj.w = nn.Parameter(self.best_w[i].to(self.device))
        
    def valT(self,valGenerator):
        self.eval()  
        with torch.no_grad():
            TLoss = None
            dataLen = 0
            for xb,missMask, yb in valGenerator:
                xb,missMask, yb = xb.type(self.dtype).to(self.device),missMask.to(self.device), yb.type(self.dtype).to(self.device)
                TLoss = self.processOneBatch('T',xb,yb,missMask,TLoss,None,None)
                dataLen += xb.size(0)
            
        return TLoss / dataLen
    
    def fit(self,trainGenerator,valGenerator,epoches,lr,weight_decay=None,
            optimizerObj=None,epoches_begin = 0):
        
        if optimizerObj is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0) 
        else:
            optimizer = optimizerObj(self.parameters(), lr=lr) 
        if self.useScheduler:
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr,
                        max_lr=lr*100,step_size_up=100,cycle_momentum=False)
        for WDDObj in self.WDDList:
            WDDObj.sampleInitPos(trainGenerator)
            
        S_epochesList = []
        ValMCC_List = []
        self.bestState_dict_path = self.outSavePath+os.sep+'bestStateDict.pt'
        
        if self.record_thrMCC:
            self.recordPath = self.outSavePath+os.sep+'recordEpoch'
            if os.path.exists(self.recordPath):
                shutil.rmtree(self.recordPath)
            os.makedirs(self.recordPath,exist_ok=True)
        
        if self.recordEpochTargets:
            self.savePathData = outSavePath + os.sep + 'dataSetCollect_epoch'
            if os.path.exists(self.savePathData):
                shutil.rmtree(self.savePathData)
            # os.makedirs(outSavePath,exist_ok=True)
            os.makedirs(self.savePathData,exist_ok=True)
                
            
        for e in range(epoches_begin,epoches):
            t_list = []
            for WDDObj in self.WDDList:
                t_list.append(WDDObj.t.detach().cpu())
                
            All_T =torch.concat(t_list,axis=0).numpy()
            S_epochesList.append(caculate_S(All_T,16))
                
            self.train()                
            optimizer.zero_grad()
            TLoss = None
            ClsLoss = None
            oriLabels = []
            predLabels = []
            for WDDObj in self.WDDList:
                WDDObj.sum_byf *= 0
                WDDObj.sum_mask *= 0
            for  xb,missMask, yb in trainGenerator:
                xb,missMask, yb = xb.type(self.dtype).to(self.device),missMask.to(self.device), yb.type(self.dtype).to(self.device)
                TLoss,ClsLoss = self.processOneBatch('N',xb,yb,missMask,TLoss,ClsLoss,[oriLabels,predLabels])

                if self.miniBatch:
                    optimizer.step()
                    if self.useScheduler:
                        scheduler.step()

            if self.distScale:
                batchDistLoss = self.computeDistLoss()
                # batchDistLoss.backward()
            else:
                batchDistLoss = 0
            DistLoss = batchDistLoss
            loss = ClsLoss + TLoss + DistLoss
            # loss.backward()
            if not self.miniBatch:
                optimizer.step()
                if self.useScheduler:
                    scheduler.step()
            if self.useScheduler:
                curr_lr = scheduler.get_last_lr()[-1]
            else:
                curr_lr = lr
            val_loss,valAcc,valMcc = self.val(valGenerator)
            ValMCC_List.append(valMcc)
            
            if self.record_thrMCC:
                if valMcc > self.record_thrMCC:
                    torch.save(self.state_dict(),self.recordPath+os.sep+'StateDictEpoch%s_MCC%.4f.pt'%(e,valMcc))  
 
            if self.bestValLoss is None:
                self.bestValLoss = val_loss
                if not self.earlyStopThres is None:
                    torch.save(self.state_dict(),self.bestState_dict_path)         
                self.bestEpoch = e
            else:
                if val_loss < self.bestValLoss:
                    self.bestValLoss = val_loss
                    if not self.earlyStopThres is None:
                        torch.save(self.state_dict(),self.bestState_dict_path)    
                    self.bestEpoch = e
                    if self.recordbestEpoch == True:
                        self.saveTargets(e)
                        trainSetCollect = self.predict(trainGenerator,returnDataSet=True)[-1]
                        valSetCollect = self.predict(valGenerator,returnDataSet=True)[-1]
                        with open(self.savePathData+os.sep+'%d.pt' %e,'wb') as FIDO:
                            pickle.dump({'tr':trainSetCollect,'val':valSetCollect},FIDO)   
            
            print('********epoch:%d, lr:%2.e*******' %(e,curr_lr))
            print(time.ctime())
            if self.distScale:
                print('Training Set: %.4f (t: %.4f + cls: %.4f + dist: %.4f):' %(loss/ len(predLabels),TLoss/ len(predLabels),ClsLoss/ len(predLabels), DistLoss) ) 
            else:
                print('Training Set: %.4f (t: %.4f + cls: %.4f):' %(loss/ len(predLabels),TLoss/ len(predLabels),ClsLoss/ len(predLabels)  ) ) 
            print('Validation Set:',val_loss,valAcc,valMcc)
            print('Curr Best:',self.bestEpoch,self.bestValLoss)
            
            if self.record_thrMCC:
                if valMcc > self.record_thrMCC:
                    if self.recordEpochTargets:
                        self.collect_and_save_data(e, valMcc, trainGenerator, valGenerator)
               
            else:
                if e % self.recordEpochStepSize == 0 and self.recordEpochTargets:
                    self.collect_and_save_data(e, valMcc, trainGenerator, valGenerator)
    
            if e - self.bestEpoch > self.earlyStopThres:
                print('Early stopped.')
                return  self.bestEpoch,S_epochesList,ValMCC_List 
            
        if e == epoches-1:
            return self.bestEpoch,S_epochesList,ValMCC_List   
            
    def val(self,valGenerator):
        self.eval()  
        with torch.no_grad():
            TLoss = None
            ClsLoss = None
            oriLabels = []
            predLabels = []
            for xb,missMask, yb in valGenerator:
                xb,missMask, yb = xb.type(self.dtype).to(self.device),missMask.to(self.device), yb.type(self.dtype).to(self.device)
                TLoss,ClsLoss = self.processOneBatch('N',xb,yb,missMask,TLoss,ClsLoss,[oriLabels,predLabels])
            ClsLoss = ClsLoss / len(predLabels)
            TLoss = TLoss / len(predLabels)
            loss = ClsLoss + TLoss 
            valAcc = accuracy_score(oriLabels,predLabels)
            valMcc = matthews_corrcoef(oriLabels,predLabels)
        return loss,valAcc,valMcc
    
    def predict(self,testGenerator,returnDataSet = False):
        self.eval()  
        if returnDataSet:
            dataSetCollect = {'x':None,'y':None,'mask':None,'y_pred':None,'pred_probability':None,'dist_prob':None}
        with torch.no_grad():
            TLoss = None
            ClsLoss = None
            oriLabels = []
            predLabels = []
            for xb,missMask, yb in testGenerator:
                xb,missMask, yb = xb.type(self.dtype).to(self.device),missMask.to(self.device), yb.type(self.dtype).to(self.device)
                TLoss,ClsLoss,PredProb,dist_probs,_ = self.processOneBatch('N',xb,yb,missMask,TLoss,ClsLoss,[oriLabels,predLabels],returnPredProb=True)
                if returnDataSet:
                    if dataSetCollect['x'] is None:
                        dataSetCollect['x'] = xb.cpu()
                    else:
                        dataSetCollect['x'] = torch.cat([dataSetCollect['x'], xb.cpu()],dim=0)
                        
                    if dataSetCollect['y'] is None:
                        dataSetCollect['y'] = yb.cpu()
                    else:
                        dataSetCollect['y'] = torch.cat([dataSetCollect['y'], yb.cpu()],dim=0)
                        
                    if dataSetCollect['mask'] is None:
                        dataSetCollect['mask'] = missMask.cpu()
                    else:
                        dataSetCollect['mask'] = torch.cat([dataSetCollect['mask'], missMask.cpu()],dim=0)
                        
                    if dataSetCollect['pred_probability'] is None:
                        dataSetCollect['pred_probability'] = PredProb.cpu()
                    else:
                        dataSetCollect['pred_probability'] = torch.cat([dataSetCollect['pred_probability'], PredProb.cpu()],dim=0)
                   
                    if dataSetCollect['dist_prob'] is None: 
                        key_list = ['class%s'%i for i in range(len(dist_probs))]
                        value_list = [dist_probs[i] for i in range(len(dist_probs))]
                        dataSetCollect['dist_prob'] = dict(zip(key_list,value_list))
                    else:
                        for tmp_key in range(len(dataSetCollect['dist_prob'].keys())):
                            dataSetCollect['dist_prob']['class%s'%tmp_key] = torch.cat([dataSetCollect['dist_prob']['class%s'%tmp_key],dist_probs[tmp_key]],dim=1)                            
                    
            ClsLoss = ClsLoss / len(predLabels)
            TLoss = TLoss / len(predLabels)
            loss = ClsLoss + TLoss
            if returnDataSet:
                dataSetCollect['x'] = dataSetCollect['x'].cpu().detach().numpy()
                dataSetCollect['y'] = torch.argmax(dataSetCollect['y'].detach().cpu(),dim=1).numpy()
                dataSetCollect['mask'] = dataSetCollect['mask'].cpu().detach().numpy()
                dataSetCollect['y_pred'] = np.array(predLabels)
                dataSetCollect['pred_probability'] = np.array(dataSetCollect['pred_probability'].cpu())
                return loss,oriLabels,predLabels,dataSetCollect
        return loss,oriLabels,predLabels
    
    
    def predict_singleSample(self,testdata,returnDataSet = False):
        self.eval()  
        if returnDataSet:
            dataSetCollect = {'x':None,'y':None,'mask':None,'y_pred':None,'pred_probability':None,'dist_prob':None}
        with torch.no_grad():
            TLoss = None
            ClsLoss = None
            oriLabels = []
            predLabels = []
            xb,missMask, yb = testdata['x'].type(self.dtype).to(self.device), testdata['missMask'].to(self.device), testdata['yb'].type(self.dtype).to(self.device)
            TLoss,ClsLoss,PredProb,dist_probs,_ = self.processOneBatch('N',xb,yb,missMask,TLoss,ClsLoss,[oriLabels,predLabels],returnPredProb=True)
            if returnDataSet:
                if dataSetCollect['x'] is None:
                    dataSetCollect['x'] = xb.cpu()
                else:
                    dataSetCollect['x'] = torch.cat([dataSetCollect['x'], xb.cpu()],dim=0)
                    
                if dataSetCollect['y'] is None:
                    dataSetCollect['y'] = yb.cpu()
                else:
                    dataSetCollect['y'] = torch.cat([dataSetCollect['y'], yb.cpu()],dim=0)
                    
                if dataSetCollect['mask'] is None:
                    dataSetCollect['mask'] = missMask.cpu()
                else:
                    dataSetCollect['mask'] = torch.cat([dataSetCollect['mask'], missMask.cpu()],dim=0)
                    
                if dataSetCollect['pred_probability'] is None:
                    dataSetCollect['pred_probability'] = PredProb.cpu()
                else:
                    dataSetCollect['pred_probability'] = torch.cat([dataSetCollect['pred_probability'], PredProb.cpu()],dim=0)
               
                if dataSetCollect['dist_prob'] is None: 
                    key_list = ['class%s'%i for i in range(len(dist_probs))]
                    value_list = [dist_probs[i] for i in range(len(dist_probs))]
                    dataSetCollect['dist_prob'] = dict(zip(key_list,value_list))
                else:
                    for tmp_key in range(len(dataSetCollect['dist_prob'].keys())):
                        dataSetCollect['dist_prob']['class%s'%tmp_key] = torch.cat([dataSetCollect['dist_prob']['class%s'%tmp_key],dist_probs[tmp_key]],dim=1)                            
            ClsLoss = ClsLoss / len(predLabels)
            TLoss = TLoss / len(predLabels)
            loss = ClsLoss + TLoss
            if returnDataSet:
                dataSetCollect['x'] = dataSetCollect['x'].cpu().detach().numpy()
                dataSetCollect['y'] = torch.argmax(dataSetCollect['y'].detach().cpu(),dim=1).numpy()
                dataSetCollect['mask'] = dataSetCollect['mask'].cpu().detach().numpy()
                dataSetCollect['y_pred'] = np.array(predLabels)
                dataSetCollect['pred_probability'] = np.array(dataSetCollect['pred_probability'].cpu())
                return loss,oriLabels,predLabels,dataSetCollect
            return loss,oriLabels,predLabels        
        
       

def processW(wIn):
    # w = torch.abs(wIn)
    # return w
    # return w / w.sum()
    # return torch.abs(wIn)
    # mu = wIn.mean()
    # std = wIn.std()
    # wIn = (wIn - mu) / (std + 1e-5)
    # wIn = 1/(1+torch.exp(-1*wIn))
    # return wIn
    # return torch.sigmoid(wIn)
    return wIn

def filterW(wIn):
    # tmp = torch.abs(wIn)
    # tmp = tmp <= torch.quantile(tmp,0.1)
    # out = wIn.masked_fill(tmp,0)
    # return out
    return wIn

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

def UMAP(data,label=None):
    import umap
    reducer = umap.UMAP( n_epochs=100,metric='euclidean')
    
    if not label is None:
        reducer.fit(data,label)
    else:
        reducer.fit(data)
    
    
    return reducer 

def processUMAPFromGenerator(trainGenerator, doSupervise=False):
    
    xtr,ytr,mtr = getAllDataSet(trainGenerator,fillNan=False)
    avg = xtr.sum(dim=0) / (~mtr).sum(dim=0)
    x2_avg = (xtr - avg.view([1,-1])) ** 2
    x2_avg.masked_fill_(mtr,0)
    std = (x2_avg.sum(dim=0) / (~mtr).sum(dim=0)) ** 0.5
    xtr = (xtr - avg.view([1,-1])) / std.view([1,-1])
    xtr.masked_fill_(mtr,0)
    if doSupervise:
        reducer = UMAP(xtr,label=ytr)    
    else:
        reducer = UMAP(xtr)
    return reducer,reducer.transform(xtr)

def normFromTrain(xtr,mtr,xva,mva,xte,mte):
    avg = xtr.sum(dim=0) / (~mtr).sum(dim=0)
    x2_avg = (xtr - avg.view([1,-1])) ** 2
    x2_avg.masked_fill_(mtr,0)
    std = (x2_avg.sum(dim=0) / (~mtr).sum(dim=0)) ** 0.5
    xtr = (xtr - avg.view([1,-1])) / std.view([1,-1])
    xtr.masked_fill_(mtr,0)
    xva = (xva - avg.view([1,-1])) / std.view([1,-1])
    xva.masked_fill_(mva,0)
    xte = (xte - avg.view([1,-1])) / std.view([1,-1])
    xte.masked_fill_(mte,0)
    return xtr,xva,xte

def performRF(trainGenerator,valGenerator,testGenerator,WDDModel=None,impute=False):
    from sklearn.ensemble import RandomForestClassifier
    xtr,ytr,mtr = getAllDataSet(trainGenerator,fillNan=impute)
    xva,yva,mva = getAllDataSet(valGenerator,fillNan=impute)
    xte,yte,mte = getAllDataSet(testGenerator,fillNan=impute)
    xtr,xva,xte = normFromTrain(xtr,mtr,xva,mva,xte,mte)
    
    
    if impute: 
        xtr = xtr.masked_fill(mtr,np.nan)
        xva = xva.masked_fill(mva,np.nan)
        xte = xte.masked_fill(mte,np.nan)
        xtr,xva,xte = xtr.numpy(),xva.numpy(),xte.numpy()
        imputer = KNNImputer(n_neighbors=15, weights="uniform")
        imputer.fit(xtr)
        xtr=imputer.transform(xtr)
        xva=imputer.transform(xva)
        xte=imputer.transform(xte)
    else:
        xtr,xva,xte = xtr.numpy(),xva.numpy(),xte.numpy()
    
    if not WDDModel is None:
        xtr = xtr - WDDModel.t.cpu().detach().numpy()
        xva = xva - WDDModel.t.cpu().detach().numpy()
        xte = xte - WDDModel.t.cpu().detach().numpy()   

    clf = RandomForestClassifier(max_depth=None, random_state=0)
    clf.fit(xtr,ytr)
    yva_pred = clf.predict(xva)
    yte_pred = clf.predict(xte)
    print('RFval')
    computeMetrics(yva_pred,yva)
    print('RFtest')
    computeMetrics(yte_pred,yte)
    return clf
    
def computeMetrics(prediction, testLabelArr, outPath=None):
    perform_dict = {}
    ACC = accuracy_score(testLabelArr,prediction)
    F1 = f1_score(testLabelArr,prediction)
    Recall = recall_score(testLabelArr,prediction)
    Pre = precision_score(testLabelArr,prediction)
    MCC = matthews_corrcoef(testLabelArr,prediction)
    cm=confusion_matrix(testLabelArr,prediction)
    perform_dict['ACC'] = ACC
    perform_dict['F1'] = F1
    perform_dict['Recall'] = Recall
    perform_dict['Pre'] = Pre
    perform_dict['MCC'] = MCC
    perform_dict['cm'] = cm.tolist()
    if not outPath is None:
        tmpCMPath = outPath + os.sep + 'performance'
        with open(tmpCMPath, 'w') as FIDO:
            FIDO.write('Confusion Matrix:\n')
            for i in range(cm.shape[0]):
                tmpStr = ''
                for j in range(cm.shape[1]):
                    tmpStr += '%d\t' %cm[i,j]
                tmpStr += '\n'
                FIDO.write(tmpStr)
            FIDO.write('Predicting Performance:\n')
            FIDO.write("ACC: %f \n"%ACC)
            FIDO.write("F1: %f \n"%F1)
            FIDO.write("Recall: %f \n"%Recall)
            FIDO.write("Pre: %f \n"%Pre)
            FIDO.write("MCC: %f \n"%MCC)
    
    print('Predicting Performance:')
    print("ACC: %f "%ACC)
    print("F1: %f "%F1)
    print("Recall: %f "%Recall)
    print("Pre: %f "%Pre)
    print("MCC: %f "%MCC)    
    print('Confusion Matrix:')
    print(cm)
    return perform_dict

def analysisDist(dataGenerator,batchSize = 512,bins = torch.arange(0,10,0.2).float(),
                 distMeasureScale=1e-1,device='cpu'):
    xtr, yOut,mtr = getAllDataSet(dataGenerator)
    xtr, mtr = xtr.to(device).float(), mtr.to(device)
    xtr.masked_fill_(mtr,0)
    avg = xtr.sum(dim=0) / (~mtr).sum(dim=0)
    x2_avg = (xtr - avg.view([1,-1])) ** 2
    x2_avg.masked_fill_(mtr,0)
    std = (x2_avg.sum(dim=0) / (~mtr).sum(dim=0)) ** 0.5
    xtr = (xtr - avg.view([1,-1])) / std.view([1,-1])
    xtr.masked_fill_(mtr,0)
    histTotal = None
    for i in range(int(np.ceil(xtr.size(0) / batchSize))):
        sPos = i * batchSize
        ePos = (i+1) * batchSize
        if ePos > xtr.size(0):
            ePos = xtr.size(0)
        tmpMat = xtr[sPos:ePos,:].clone()
        
        squaredDistMat = ((tmpMat.unsqueeze(1) - xtr.unsqueeze(0))**2).sum(dim=-1)
        
        tmpEyeMask = torch.zeros_like(squaredDistMat,dtype=bool,device=device)
        tmpEyeMask[torch.arange(tmpMat.size(0)),torch.arange(sPos,sPos+tmpMat.size(0))] = 1
        
        squaredDistMat.masked_fill_(tmpEyeMask,1e4)
        
        hist,bins = torch.histogram(squaredDistMat.min(dim=1).values.cpu().float(),bins=bins)
        if histTotal is None:
            histTotal = hist
        else:
            histTotal += hist #no grad!!! ha ha ha
    bestH = 0
    bestU = None
    bestB = None
    for i in range(histTotal.size(0)):
        h = histTotal[i]
        u = bins[i]
        b = bins[i+1]
        print('%.1f -- %.1f: %d' %(u,b,h))
        if h > bestH:
            bestU = u
            bestB = b
            bestH = h
    r2 = (bestU + bestB) / 2
    return 1 / distMeasureScale / r2
    
    

def distPair(attIn):
    diff = attIn.unsqueeze(0) - attIn.unsqueeze(1)
    diff = (diff ** 2).sum(dim=-1) ** 0.5
    return diff

def spDist(x,t,w,mask,gamma,delta):
    tmpW = processW(w)
    diff = t.unsqueeze(0) - x # n * f
    if not mask is None:
        diff = diff.masked_fill(mask.unsqueeze(0),0)
    dist = ((tmpW * diff ) ** 2).sum(axis=-1) # n
    prob = torch.exp(-1*gamma*dist/ ((x.size(1) - mask.sum(dim=1) + 1)**delta) )
    return prob

def filterDiscWeight(tmpDiscW, accuThres=0.95):
    tmpDiscW = tmpDiscW.detach().clone()
    tmpFlatten = tmpDiscW.flatten()
    tmpSortObj = (tmpFlatten.abs() / tmpFlatten.abs().sum()).sort(descending=True)
    accuW = 0.
    i = 0
    while accuW < accuThres:
        accuW += tmpSortObj.values[i].item()
        i += 1
    absThresValue = tmpFlatten[tmpSortObj.indices[i - 1]].abs()
    tmpDiscW[tmpDiscW.abs()<absThresValue] = 0
    return tmpDiscW

def filterDiscWeight_numRatio(tmpDiscW, numRatio=0.95):
    tmpDiscW = tmpDiscW.detach().clone()
    tmpFlatten = tmpDiscW.flatten()
    tmpSortObj = tmpFlatten.abs().sort(descending=True)
    num = int(len(tmpSortObj.values)*numRatio)
    absThresValue = tmpSortObj.values[num].abs()
    tmpDiscW[tmpDiscW.abs()<absThresValue] = 0
    return tmpDiscW




def traverse_folders(root_folder):
    csv_files = [] 
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    return csv_files
 
def weight_analysis(my_model,label_num,feature_name,com_mod,acc_filter_discW,outSavePath):
    discWeight_list = []
    t_list = []
    w_list = []
    significance_list = []
    for i in range(label_num):
        tmp_discWeight = filterDiscWeight(my_model.discLayers[i].linear.weight,acc_filter_discW)    
        discWeight_list.append(tmp_discWeight)
        t_list.append(my_model.WDDList[i].t.clone())
        w_list.append(my_model.WDDList[i].w.clone())
        if com_mod == 'tanh':
            filter_wdd_W = torch.abs(torch.tanh(torch.abs(tmp_discWeight.t()))*torch.tanh(torch.abs(my_model.WDDList[i].w.clone())))
        elif com_mod =='log':
            filter_wdd_W = torch.log(1+torch.abs(tmp_discWeight.t())*torch.log(1+torch.abs(my_model.WDDList[i].w.clone())))
        else:
            filter_wdd_W = torch.abs(pow(torch.abs(tmp_discWeight.t()),1/len(feature_name))*torch.exp(-(1*my_model.WDDList[i].gamma/((torch.abs(my_model.WDDList[i].w.clone())))))) 
        significance_list.append(filter_wdd_W)  
    df_sum_WDD_w_all_sort = ana_W(significance_list,feature_name,outSavePath)
    return df_sum_WDD_w_all_sort
        
    
def ana_W(significance_list,feature_names,outSavePath):
    import matplotlib.pyplot as plt

    df_sum_WDD_w_list = []
    df_sum_WDD_w_all = None
    for i in range(len(significance_list)):
        sum_WDD_w = significance_list[i].sum(axis=0).detach().cpu()
        df_sum_WDD_w = pd.DataFrame(sum_WDD_w,index=feature_names)
        df_sum_WDD_w_list.append(df_sum_WDD_w)
        if df_sum_WDD_w_all is None:
            df_sum_WDD_w_all = df_sum_WDD_w
        else:
            df_sum_WDD_w_all = df_sum_WDD_w_all+df_sum_WDD_w.copy()
    df_sum_WDD_w_all_sort = df_sum_WDD_w_all.sort_values(by=0,ascending=True)
    
    plt.rcParams['figure.dpi'] = 600  
    plt.rcParams['font.sans-serif'] = 'Arial'  
    fig, ax = plt.subplots(dpi=600)
    for idx,df_sum_WDD_w_class in enumerate(df_sum_WDD_w_list):
        
        df_class = df_sum_WDD_w_class.reindex(df_sum_WDD_w_all_sort.index)

        if idx == 0:
            ax.barh(df_sum_WDD_w_all_sort.index, df_class[0], label='Class%s'%idx,color='blue')
            df_class_old = df_class[0]
        else:
            ax.barh(df_sum_WDD_w_all_sort.index, df_class[0], left=df_class_old,label='Class%s'%idx,color='blue')
            df_class_old = df_class_old+df_class[0]
            
    ax.legend()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=5)
    plt.savefig(outSavePath+os.sep+'ana_W.png', dpi=600)
    return df_sum_WDD_w_all_sort
    
    
def ana_market(SetCollect,mymodel,top_k=None,acu=None,dpi=600):
    import matplotlib.pyplot as plt
    layerpro_list = []
    layerprob_mul_discw_list  = []
    disc_w_list = []
    for idx in range(len(SetCollect['dist_prob'].keys())):
        tmp_class_w = mymodel.discLayers[idx].weight.detach().cpu().numpy()
        class_layerprob =  SetCollect['dist_prob']['class%s'%idx].detach().cpu().numpy()
        sorted_indices = np.argsort(tmp_class_w.flatten())
        class_layerprob_sort = class_layerprob[sorted_indices]
        class_layerprob_mul_discw = class_layerprob* tmp_class_w.transpose()
        class_layerprob_mul_discw_sort =  class_layerprob_mul_discw[sorted_indices]
        layerprob_mul_discw_list.append(class_layerprob_mul_discw_sort)
        layerpro_list.append(class_layerprob_sort)
        disc_w_list.append(tmp_class_w)
        
    all_layerprob_mul_discw = np.concatenate(layerprob_mul_discw_list,axis=0) 

    for i in layerprob_mul_discw_list:
        if top_k:
            top_k_metrics,overlap_matrix = top_k_overlap(i,k=top_k,dig=True)
        if acu:
            con_p_metrics,overlap_matrix =  contribution_percent_overlap(torch.tensor(i),acu,dig=True)
            
        plt.imshow(overlap_matrix, cmap='Reds', interpolation='nearest')
        plt.colorbar()
        plt.show(dpi)
        
    if top_k:
        top_k_metrics,overlap_matrix =  top_k_overlap(all_layerprob_mul_discw,k=top_k,dig=True)
    if acu:
        con_p_metrics,overlap_matrix =  contribution_percent_overlap(torch.tensor(all_layerprob_mul_discw),acu,dig=True)
    plt.imshow(overlap_matrix, cmap='Reds', interpolation='nearest')
    plt.colorbar()
    plt.show(dpi)
        
def top_k_overlap(prob_matrix,k,dig =False):
    top_k = np.zeros_like(prob_matrix)
    for i in range(prob_matrix.shape[1]):
        top_k_indices = np.argsort(prob_matrix[:, i])[-k:]
        top_k[top_k_indices, i] = 1
    if dig:
        mul= top_k@(top_k.transpose())
        overlap = np.divide( mul,top_k.sum(axis=1)[np.newaxis,:].transpose(),out =np.zeros_like(mul) ,where = top_k.sum(axis=1)[np.newaxis,:].transpose()!=0)
    else:
        overlap = top_k@(top_k.transpose())
    return top_k,overlap     

def top_k_overlap_byt(prob_matrix,k):
    top_k = np.zeros_like(prob_matrix)
    for i in range(prob_matrix.shape[0]):
        top_k_indices = np.argsort(prob_matrix[i, :])[-k:]
        top_k[i, top_k_indices] = 1
   
    overlap = top_k@(top_k.transpose())
    return top_k,overlap    

def filterWDD_W(wdd_W,accuThres=0.95):
    abs_wdd_W=torch.abs(wdd_W).detach().cpu()
    conlumns_sums = torch.sum(abs_wdd_W, dim=0)
    sorted_wdd, _ = torch.sort(abs_wdd_W,dim=0,descending=True)
    sorted_wdd_ratio = sorted_wdd/conlumns_sums.unsqueeze(0)
    thre_list = []
    for i in range(sorted_wdd_ratio.shape[1]):
        tmp = sorted_wdd_ratio[:,i]
        accuW = 0.
        j = 0
        while accuW < accuThres:
            accuW += tmp[j].item()
            j += 1
        absThresValue_idx = j-1
        thre_list.append(sorted_wdd[absThresValue_idx,i])
    mask = torch.where(abs_wdd_W<torch.tensor(thre_list).unsqueeze(0), torch.zeros_like(abs_wdd_W), torch.ones_like(abs_wdd_W))
    masked_tensor = abs_wdd_W * mask
    return  masked_tensor.numpy(),mask.numpy()

def contribution_percent_overlap(prob_matrix,accThres,dig=False):
    _,con_p = filterWDD_W(prob_matrix,accThres)
    if dig:
        mul= con_p @(con_p.transpose())
        overlap = np.divide(mul,con_p.sum(axis=1)[np.newaxis,:].transpose(),out =np.zeros_like(mul) ,where = con_p.sum(axis=1)[np.newaxis,:].transpose()!=0)
    else:
        overlap = con_p @(con_p.transpose())
    return con_p, overlap

def collate_fn(batch, extra_param=None):
    selected_features_list = [item_data[0][extra_param].unsqueeze(0) for item_data in batch]
    selected_mask_list = [item_data[1][extra_param].unsqueeze(0) for item_data in batch]
    selectedY_list = [item_data[2].unsqueeze(0) for item_data in batch]

    selectedF = torch.cat(selected_features_list, dim=0)
    selectedM = torch.cat(selected_mask_list, dim=0)
    selectedY = torch.cat(selectedY_list, dim=0)
    
    selected_batches = [selectedF, selectedM, selectedY]
    return selected_batches


def caculate_S(data,bins_num=16):
    n, f = data.shape
    q_values = np.percentile(data, q=[25, 50, 75], axis=0)
    iqr_values = q_values[2] - q_values[0]

    bin_edges = np.zeros((f, bins_num + 1))
    for i in range(f):
        lower_bound = q_values[0, i] - 1.5 * iqr_values[i]
        upper_bound = q_values[2, i] + 1.5 * iqr_values[i]
        bin_edges[i, :] = np.linspace(lower_bound, upper_bound, bins_num+1)
        
    binned_data = np.zeros((n, f, bins_num +2), dtype=int)
    for i in range(f):
        for j in range(bins_num +2):
            if j == 0:
                binned_data[:, i, j] = (data[:, i] < bin_edges[i, j])
            elif j == bins_num+1:
                binned_data[:, i, j] = (data[:, i] >= bin_edges[i, j-1])
            else:
                lower_bound = bin_edges[i, j-1]
                upper_bound = bin_edges[i, j]
                binned_data[:, i, j] = ((data[:, i] >= lower_bound) & (data[:, i] < upper_bound))
                
    binned_data_sum = np.sum(binned_data,axis=0)
    binned_data_norm=(binned_data_sum/np.expand_dims(binned_data_sum.sum(1),1))+1e-30
    entropy_list = []

    for i in range(f):
        tmp_data = binned_data_norm[i,:]
        entropy =  -np.sum(tmp_data*np.log2(tmp_data))
        entropy_list.append(entropy)
    return entropy_list


def plot_stacked_feature_importance_cumulative(model_weights_list, feature_names, cumulative_threshold, outSavePath=None,plot=True):
    import matplotlib.pyplot as plt
    num_models = len(model_weights_list)
    num_features = model_weights_list[0].size(1)
    
    feature_importance_matrix = torch.zeros(num_models, num_features)
    for i in range(num_models):
        abs_weights = torch.abs(model_weights_list[i])
        sorted_indices = torch.argsort(abs_weights, dim=1, descending=True)
        sorted_abs_weights = torch.gather(abs_weights, 1, sorted_indices)
        cumulative_sum = torch.cumsum(sorted_abs_weights, dim=1)
        total_sum = torch.sum(abs_weights, dim=1)  # Corrected line
        
        important_indices = torch.where(cumulative_sum <= cumulative_threshold * total_sum.view(-1, 1), 1, 0)
        binary_matrix = torch.zeros_like(abs_weights, dtype=torch.float)  # Use dtype=torch.float
        binary_matrix[torch.arange(important_indices.shape[0]).unsqueeze(1), sorted_indices] = important_indices.float()  # Convert to float
        feature_importance_matrix[i] = torch.sum(binary_matrix, dim=0)
    
    total_feature_importance = torch.sum(feature_importance_matrix, dim=0)
    sorted_indices = torch.argsort(total_feature_importance, descending=False)
    
    if plot:
        plt.figure(figsize=(10, 8))
        bottom = torch.zeros(num_features)
        for i in range(num_models):
            importance_vector = feature_importance_matrix[i, sorted_indices]
            plt.barh(range(num_features), importance_vector, left=bottom, label=f"Class {i+1}")
            bottom += importance_vector
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Stacked Feature Importance')
        plt.legend()
        plt.yticks(range(num_features), [feature_names[i] for i in sorted_indices])
        if not outSavePath is None:
            plt.savefig(outSavePath+os.sep+'anaWbyAccum.png',dpi=600)
        plt.show()
    
    tmp_f=[feature_names[i] for i in sorted_indices]
    tmp_f.reverse()
    return feature_importance_matrix[:, sorted_indices],tmp_f



def distDi(x,t,w,m=None):
    diff = t.unsqueeze(1) - x.unsqueeze(0) # nt * n * f
    tmpW = processW(w).unsqueeze(1)
    if not m is None:
        diff = diff.masked_fill(m.unsqueeze(0),0)
    tmp_diff = (tmpW * diff ) ** 2 
    return tmp_diff


        
        

        
