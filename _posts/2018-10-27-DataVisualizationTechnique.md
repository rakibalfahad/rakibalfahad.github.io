---
title: "Data visualization Technique"
date: 2018-10-27
tags: [Data visualization, Higher dimensional data, tSNE, Manifold]
header:
  #image: "/images/KerasTensorflow.jpg"
excerpt: "Data visualization, Higher dimensional data, tSNE, Manifold"
mathjax: "true"
---


```python
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs
#-----------My utility codes-------------------------
import sys
sys.path.append('/home/ralfahad/MyMac/UtilityCodes')
from LDAAnalysis import LDA
#-----------My utility codes-------------------------
# for jumpyter notebook
%matplotlib inline
# Generate a random dataset with 1000 dimention and 100 smaple which represent a higher dimentional data
X, y = make_blobs(n_samples=100, centers=2, n_features=1000, random_state=0)

# Dimentionality reduction on PCA
pca = PCA(n_components=50)
X_PCA=pca.fit_transform(X)

# to increase inter class difference we apply LDA
X_PCA_LDA=LDA(X_PCA, y, 50)[0]
# Apply tSNE
tsne = TSNE(n_components=2, random_state=0, perplexity=10)
Y = tsne.fit_transform(X_PCA_LDA) # this

from tSNE_KDE_2Class_VisPlot import tSNE_KDE_2Class_VisPlot # import function from my utility
tSNE_KDE_2Class_VisPlot(Y,y, fileName="tSNE_KDE_plot")

```
Image generated from this code:
![alt]({{ site.url }}{{ site.baseurl }}/images/DataVisualization/tSNE_KDE_plot.png)
