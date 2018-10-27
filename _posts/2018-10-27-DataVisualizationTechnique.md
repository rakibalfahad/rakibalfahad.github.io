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

a=Y[np.where(y==0), 0]
b=Y[np.where(y==0), 1]
c= Y[np.where(y==1), 0]
d = Y[np.where(y==1), 1]
dftt1=pd.concat([pd.DataFrame(a),pd.DataFrame(b)], axis=0).transpose()
dftt2=pd.concat([pd.DataFrame(c),
            pd.DataFrame(d)], axis=0).transpose()
dftt1.shape,dftt2.shape
dftt1.columns= ['a','b']
dftt2.columns= ['c','d']
ax = sns.kdeplot(dftt1['a'],dftt1['b'], cmap="Greens", shade=False, shade_lowest=False)
ax = sns.regplot(dftt1['a'], dftt1['b'],marker= '.', color='g',  fit_reg=False,label='Class_1')
ax.legend(loc="best",framealpha=0.0)
ax = sns.kdeplot(dftt2['c'],dftt2['d'],cmap="Reds", shade=False, shade_lowest=False)
ax = sns.regplot(dftt2['c'], dftt2['d'],marker= '+', color='r',  fit_reg=False, label='Class_2')
ax.legend(loc="best",framealpha=0.0)
# Add labels to the plot
red = sns.color_palette("Greens")[-2]
blue = sns.color_palette("Reds")[-2]
# Save image
save_format='png'
fileName="tSNE_KDE_plot"
print str(fileName)+'.'+save_format
plt.savefig(fileName+'.'+save_format,dpi=300)
```
Image generated from this code:
![alt]({{ site.url }}{{ site.baseurl }}/images/DataVisualization/tSNE_KDE_plot.png)
