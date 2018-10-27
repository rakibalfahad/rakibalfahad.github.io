---
title: "High Dimensional Data visualization using tSNE"
date: 2018-10-27
tags: [Data visualization, Higher dimensional data, tSNE, Manifold]
header:
  #image: "/images/KerasTensorflow.jpg"
excerpt: "Data visualization, Higher dimensional data, tSNE, Manifold"
mathjax: "true"
---
# t-SNE (TSNE)
t-SNE (TSNE) converts affinities of data points to probabilities. The affinities in the original space are represented by Gaussian joint probabilities and the affinities in the embedded space are represented by Student’s t-distributions. This allows t-SNE to be particularly sensitive to local structure and has a few other advantages over existing techniques:

- Revealing the structure at many scales on a single map
- Revealing data that lie in multiple, different, manifolds or clusters
- Reducing the tendency to crowd points together at the center

While Isomap, LLE and variants are best suited to unfold a single continuous low dimensional manifold, t-SNE will focus on the local structure of the data and will tend to extract clustered local groups of samples as highlighted on the S-curve example. This ability to group samples based on the local structure might be beneficial to visually disentangle a dataset

The Kullback-Leibler (KL) divergence of the joint probabilities in the original space and the embedded space will be minimized by gradient descent. Note that the KL divergence is not convex, i.e. multiple restarts with different initializations will end up in local minima of the KL divergence. Hence, it is sometimes useful to try different seeds and select the embedding with the lowest KL divergence.

The disadvantages to using t-SNE are roughly:

- t-SNE is computationally expensive, and can take several hours on million-sample datasets
- where PCA will finish in seconds or minutes
- The Barnes-Hut t-SNE method is limited to two or three dimensional embeddings.

The algorithm is stochastic and multiple restarts with different seeds can yield different embeddings. However, it is perfectly legitimate to pick the embedding with the least error.
Global structure is not explicitly preserved. This is problem is mitigated by initializing points with PCA (using init=’pca’).

#  Tips on practical use
- Make sure the same scale is used over all features. Because manifold learning methods are based on a nearest-neighbor search, the algorithm may perform poorly otherwise. use following code

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scale=scaler.fit_transform(X) # X is the data matrix
```
- The reconstruction error computed by each routine can be used to choose the optimal output dimension. For a -dimensional manifold embedded in a -dimensional parameter space, the reconstruction error will decrease as n_components is increased until n_components == d.
- Note that noisy data can “short-circuit” the manifold, in essence acting as a bridge between parts of the manifold that would otherwise be well-separated. Manifold learning on noisy and/or incomplete data is an active area of research.
- Certain input configurations can lead to singular weight matrices, for example when more than two points in the dataset are identical, or when the data is split into disjointed groups. In this case, solver='arpack' will fail to find the null space. The easiest way to address this is to use solver='dense' which will work on a singular matrix, though it may be very slow depending on the number of input points. Alternatively, one can attempt to understand the source of the singularity: if it is due to disjoint sets, increasing n_neighbors may help. If it is due to identical points in the dataset, removing these points may help.



Reference and usefull links:

1. sklearn guide: [Here](http://scikit-learn.org/stable/modules/manifold.html#t-sne)
2. How to Use t-SNE Effectively: [Here](https://distill.pub/2016/misread-tsne/)

Example Code:
```python
"""
Author Rakib Al-Fahad
Date: 10/27/2018
This code is to give example of tSNE visualization
"""
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
