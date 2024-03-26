# DSpaLaRefiner
A data space landmark refiner deep learning algorithm, which serves as an interpretable approach employed for the construction of auxiliary diagnostic model.

![Figure1](https://github.com/Lvxiang713/DspaLaRefiner/assets/119480930/f9e27b8e-bfb5-455a-99c8-14e30b8cedec)




## 1.Brief Introduction
The basic concept of DSpaLaRefiner is to generate multiple pseudo data points, named Landmarks, in data space, and make the classification using the distance between the landmarks and the data. Inspired by the concept of SVM, gaussian radial basis function is employed as the distance measurement. The distances are used as the references for the final classification. The Landmarks act similar role with support vectors in SVM, but the positions and related weights of landmarks can be optimized by deep learning optimizers using the cross-entropy loss function, which gives DSpaLaRefiner the ability to refine the pseudo points to Landmarks. This document is a guide for users to get touch with ‘DSpaLaRefiner’ for modeling and analysis of electronic health records(EHRs).
## 2.Installation
All the code of 'DSpaLaRefiner' is wrote in Python, and no mixture code (e.g. C/C++) is used in this project, so the installation is very easy. Once the dependencies are resolved, the only thing to do is to make the path as a working path or put the code into the search path. We recommend using a Conda environment to run this algorithm. For GPU inference, the environment requires CUDA 11.7, and the pytorch version is 2.0.1. If you use cpu, it will be a bit slower, but should work.
### basic configuration
You should create a conda enviroment first.  
<pre>
conda create -n DRtorch python=3.9  
conda activate DRtorch
</pre>
Install the requirements, and it will take several minutes.  
```pip install -r requirements.txt```  
Congratulations!👍 the basic configuration is now complete.
## 3.Quick Start
### 3.1 Training step
If you installed the conda environment, you can now traning using a command line window (or terminal in Linux). Then runing:
<pre>
python run_mymodel.py --help
</pre>
if the help document is showed without error, it’s available. Users can then perform a shot test:
<pre>
optional arguments:
  -h, --help            show this help message and exit
  --dataPath DATAPATH   dataSet Path for training
  --outSavePath OUTSAVEPATH
                        Output file path
  --device DEVICE       your device, cpu, gpu or others
  --batchSize BATCHSIZE
  --num_workers NUM_WORKERS
  --recordEpochTargets  Run recordEpochTargets or not, for drawing UMAP.The output files are quite large,  we recommend to use this option only when needed for analysis.
  --recordEpochStepSize RECORDEPOCHSTEPSIZE
                        When recordEpochTargets is True, defining the step size of the recorded data
  --performWDDOpt       Run WDDloss or not.
  --WDDlossRate WDDLOSSRATE
                        The rate of WDDloss
  --tRatio TRATIO       A parameter for model,the number of target points over the number of training data
  --epoches EPOCHES     Max epoches for training
  --lr_init LR_INIT     Init learning rate for training
  --saveDataset         Save the dataset or not, for analysis.
  --gamma GAMMA         A init parameter for model, no need to change
  --earlyStopThres EARLYSTOPTHRES
                        The earlyStopThres indicates that if the validation loss remains unchanged for earlyStopThres
                        consecutive epochs, the model will trigger early stopping.
  --Mode {train,eval}
  --paraDictPath PARADICTPATH
                        Use this parameter when the mode is eval,paraDict path for initializing your model
  --StatdictPath STATDICTPATH
                        Use this parameter when the mode is eval, state dict path of your model
  --DataSetPath DATASETPATH
                        Use this parameter when the mode is eval,testdata path
</pre>  
After reading the parameter overview above, you can customize the parameters for training. For example:  
```python run_mymodel.py --dataPath ./SplitData/data_hospital_11_0.pt --outSavePath ./yourOutPath --device cuda:0 --batchSize 512 --performWDDOpt --WDDlossRate 0 --tRatio 1 --epoches 5000 --lr_init 0.001 --earlyStopThres 100 --Mode train```    
The required data format for training is presented in the ./SplitData/requiredDataFormat.ipynb.  

### 3.2 Predicting step  
After the training is complete, the model parameter files will be saved to outSavePath. If you wish to use this model for prediction, you only need to execute a single command:  
```python run_mymodel.py --device cuda:0 --batchSize 512 --Mode eval --paraDictPath ./yourOutPath/paraDict.pt --StatdictPath ./yourOutPath/bestStateDict.pt --DataSetPath  ./OutPath/testDataSet.pt```  

## 4 Interpretability analysis
The visualization code for the interpretability analysis section is stored in the 'Analysis' folder. It includes visualizations of the model training optimization process and interpretability analysis at the sample and feature levels.

## 5 Conclusion
This document is provided for users to know DSpaLaRefiner. As an open source algorithm, we have documented all the code and the functio  For more details, please see the word file 'manual.docx'.  We looking forward to receiving any bug reports and suggestions, please feel free to contact us anytime. (LX7381@gmail.com)


