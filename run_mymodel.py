# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:09:25 2023
@author:LvXiang, JingRunyu
"""

import numpy as np
import torch
import re, copy, os, time
from torch import nn
from sklearn import metrics
import pickle
import shutil
import pandas as pd
import DSpaLaRefinerModel as Mymodel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, help = 'dataSet Path for training')
    parser.add_argument('--outSavePath', type=str,help ='Output file path')
    parser.add_argument('--device', type=str, default = 'cpu', help = 'your device, cpu, gpu or others')
    parser.add_argument('--batchSize', type=int, default = 512)
    parser.add_argument('--num_workers', type=int, default = 0)
    parser.add_argument('--recordEpochTargets', action="store_true", help="Run recordEpochTargets or not, for drawing UMAP")
    parser.add_argument('--recordEpochStepSize', type=int, default = 20, help = 'When recordEpochTargets is True, defining the step size of the recorded data')
    parser.add_argument('--performWDDOpt', action="store_true", help="Run WDDloss or not.")
    parser.add_argument('--WDDlossRate', type=float, default = 0.1, help="The rate of WDDloss")
    parser.add_argument('--tRatio', type=float, default = 0.15, help='A parameter for model,the number of target points over the number of training data')
    parser.add_argument('--epoches', type=int, default =5000, help='Max epoches for training')
    parser.add_argument('--lr_init', type=float, default =1e-3, help='Init learning rate for training')
    parser.add_argument('--saveDataset', action="store_true", help="Save the dataset or not.")
    parser.add_argument('--gamma', type=float, default =2.**0, help='A init parameter for model, no need to change')
    parser.add_argument('--earlyStopThres', type=int, default =100,help='The earlyStopThres indicates that if the validation loss remains unchanged for earlyStopThres consecutive epochs, the model will trigger early stopping.')
    parser.add_argument('--Mode', type=str,default ='train',choices=['train','eval'])
    parser.add_argument('--paraDictPath', type=str, help = 'Use this parameter when the mode is eval,paraDict path for initializing your model')
    parser.add_argument('--StatdictPath',type=str,help = 'Use this parameter when the mode is eval, state dict path of your model')
    parser.add_argument('--DataSetPath',type=str,help = 'Use this parameter when the mode is eval,testdata path')
    args = parser.parse_args()
    
    
    device = args.device
    batchSize = args.batchSize
    num_workers = args.num_workers
    if args.Mode == 'train':
        print('tRatio:%s,WDDlossRate%s'%(args.tRatio,args.WDDlossRate))
        ##prepare data
        recordEpochTargets = args.recordEpochTargets
        dataPath = args.dataPath
        outSavePath = args.outSavePath
        tRatio = args.tRatio
        if os.path.exists(outSavePath):
            shutil.rmtree(outSavePath)
        os.makedirs(outSavePath,exist_ok=True)
        if args.WDDlossRate == 0:
           args.performWDDOpt = False
        a = time.time()
        with open(dataPath,'rb') as FOUT:
            tmpDataSet = pickle.load(FOUT)
        b = time.time()-a
        print('read_cost_time:%d'%b)
        trainDataSet = tmpDataSet[0]
        valDataSet = tmpDataSet[1]
        testDataSet = tmpDataSet[2]
        tmpData = trainDataSet[0][0]
        TrainGenerator = torch.utils.data.DataLoader(trainDataSet, batch_size=batchSize,shuffle=True,num_workers=num_workers)
        ValGenerator = torch.utils.data.DataLoader(valDataSet, batch_size=batchSize,shuffle=False,num_workers=num_workers)
        TestGenerator = torch.utils.data.DataLoader(testDataSet, batch_size=batchSize,shuffle=False,num_workers=num_workers)
        
        with open(outSavePath+os.sep+'testDataSet.pt','wb') as FIDO:
            pickle.dump(testDataSet,FIDO)
        
        if recordEpochTargets:
            reducer_sup,x_trans_sup = Mymodel.processUMAPFromGenerator(TrainGenerator,doSupervise=True)
            reducer_unsup,x_trans_unsup = Mymodel.processUMAPFromGenerator(TrainGenerator,doSupervise=False)
            with open(outSavePath+os.sep+'reducer.pt','wb') as FIDO:
                pickle.dump((reducer_sup,x_trans_sup,reducer_unsup,x_trans_unsup),FIDO)
                
        if not device == 'cpu':
            torch.cuda.empty_cache()
        tmpX,tmpY,tmpMask = Mymodel.getAllDataSet(TrainGenerator,fillNan=False,returnOneHotY=True)
        numT = list((tmpY.sum(dim=0) * tRatio).round().type(torch.int).numpy())
        paraDict = {
            'input_features':tmpData.size(0),
            'numClass':tmpY.size(1),
            'gamma':args.gamma,
            'device' : device,
            'dtype':torch.float32, 
            'earlyStopThres' : args.earlyStopThres,
            'computeMode':'Di',
            'batchSize':batchSize,
            'numT' : numT,
            'dropoutRate' : False,
            'distScale': None,
            'performWDDOpt': args.performWDDOpt,
            'lastAct': nn.Softmax(-1),
            'recordEpochTargets': recordEpochTargets,
            'recordEpochStepSize': args.recordEpochStepSize,
            'outSavePath':outSavePath,
            'WDDlossRate':args.WDDlossRate
            }
        
        lr = args.lr_init
        model = Mymodel.WrappedModel(**paraDict).float().to(device)
        model.processStatistic(TrainGenerator)
        with open(outSavePath+os.sep+'sta_info.pt','wb') as FIDO:
            pickle.dump((model.data_avg.detach().cpu(),model.data_std.detach().cpu()),FIDO)
        
        ##Training process
        epoches = args.epoches
        best_epoch, S_epochesList,Val_MCC  = model.fit(TrainGenerator, ValGenerator, epoches, lr,optimizerObj=torch.optim.Rprop)
        
        with open(outSavePath+os.sep+'S.pt','wb') as FIDO:
            pickle.dump(S_epochesList,FIDO)
        with open(outSavePath+os.sep+'Val_MCC.pt','wb') as FIDO:
            pickle.dump(Val_MCC,FIDO)    
        model_pre = Mymodel.WrappedModel(**paraDict).float().to(device)
        model_pre.load_state_dict(torch.load(model.bestState_dict_path))
        perform_dict_all = {}
        loss,oriLabels,predLabels = model_pre.predict(TrainGenerator)
        print('TrainSet perform:')
        perform_dict_all['train'] = Mymodel.computeMetrics(predLabels,oriLabels)
        print('------')
        print('------')
        loss,oriLabels,predLabels = model_pre.predict(ValGenerator)
        print('ValSet perform:')
        perform_dict_all['val'] = Mymodel.computeMetrics(predLabels,oriLabels)
        print('------')
        print('------')
        loss,oriLabels,predLabels = model_pre.predict(TestGenerator)
        print('TestSet perform:')
        perform_dict_all['test'] = Mymodel.computeMetrics(predLabels,oriLabels)
        print('------')
        print('------')
        with open(outSavePath+os.sep+'perform.pt','wb') as FIDO:
            pickle.dump(perform_dict_all,FIDO)
        
        with open(outSavePath+os.sep+'paraDict.pt','wb') as FIDO:
            pickle.dump(paraDict,FIDO)
        
        if args.saveDataset:
            trainSetCollect = model_pre.predict(TrainGenerator,returnDataSet=True)[-1]
            valSetCollect = model_pre.predict(ValGenerator,returnDataSet=True)[-1]
            testSetCollect = model_pre.predict(TestGenerator,returnDataSet=True)[-1]
            with open(outSavePath+os.sep+'dataSetCollect_pre.pt','wb') as FIDO:
                pickle.dump((trainSetCollect,valSetCollect,testSetCollect),FIDO)
    ##For eval
    elif args.Mode == 'eval':
        with open(args.paraDictPath,'rb') as FOUT:
            paraDict_eval = pickle.load(FOUT)
        model_pre = Mymodel.WrappedModel(**paraDict_eval).float().to(args.device)
        model_pre.load_state_dict(torch.load(args.StatdictPath))
        
        with open(args.DataSetPath,'rb') as FOUT:
            DataSet = pickle.load(FOUT)
        Datagenerator =  torch.utils.data.DataLoader(DataSet, batch_size=batchSize,shuffle=False,num_workers=num_workers)
        loss,oriLabels,predLabels = model_pre.predict(Datagenerator)
        perform = Mymodel.computeMetrics(predLabels,oriLabels)
