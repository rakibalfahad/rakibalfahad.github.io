---
title: "Importent python utility code"
date: 2018-10-27
tags: [Python, Helper function, Python Tips]
header:
  #image: "/images/KerasTensorflow.jpg"
excerpt: "Python, Helper function, Python Tips"
mathjax: "true"
---
# Description
In this page I will try to give you some helpful python code blocks that we need
to use for data preprocessing, machine learning and data visualization. I have
collected them over the time from different sources and help files.

**If lots of error/warning shows in jupyter notebook. you can stop showing
warning massages**

```python
import warnings
warnings.filterwarnings('ignore')
```

**import utility code that saved in your drive**

```python
import sys
sys.path.append('/home/ralfahad/MyMac/UtilityCodes')
# Example
from LDAAnalysis import LDA
LDA(X,y,50)
```

**Save model and data in desk with pickle formate**

```python
import pickle
DM1={'DataMatDiagonal':DataMatDiagonal
      } # make dictionary of the datasets
f = open('DataMatDiagonal.pckl', 'wb')
pickle.dump(DM1, f)
f.close()
# Load the data/model
f = open('DataMatDiagonal.pckl', 'rb')
DM1 = pickle.load(f)
f.close()
DataMatDiagonal=DM1['DataMatDiagonal']
```
**Remove a module form running jupyter notebook**
For example you change your utility code in local drive. If you don't restart the
kernel, you would not be able to access the changed function unless you delete the
module. To delete the module use the code example
    
```python
import sys
del sys.modules["PlotPCAVarienceExplanied"] # Module name PlotPCAVarienceExplanied
```
