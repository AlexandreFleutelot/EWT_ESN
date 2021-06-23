import pandas as pd
from datetime import datetime
import numpy as np

from time import time

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt
from random import sample

def load_from_csv(filepath: str):

    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M')

    df = pd.read_csv(filepath,
                     sep='\t',
                     names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                     header=None,
                     index_col='DateTime',
                     parse_dates=['DateTime'], date_parser=dateparse)

    return df

def get_mp(df, N_STEP=20):

    MP = [0] * N_STEP
    if(len(df)>1):
        maxi = max(df['High'])
        mini = min(df['Low'])
        tot_vol = sum(df['Volume'])

        for index, row in df.iterrows():
            h_idx = int(N_STEP * (row["High"] - mini) / (maxi - mini))
            l_idx = int(N_STEP * (row["Low"] - mini) / (maxi - mini))
            c_idx = h_idx - l_idx
            for i in range(l_idx, h_idx):
                MP[i] += (row["Volume"] / tot_vol / c_idx)

    #for i in range(N_STEP):
        #print('%0.5f' % (mini + (maxi-mini)*(i/10)),'\t', MP[i])

    return MP

df = load_from_csv('EURUSD60.csv')

daterange = pd.date_range('2015-01-01' , '2021-01-01')

dataset = []
for i in range(len(daterange)-1):
    start = daterange[i].strftime("%Y-%m-%d")
    end = daterange[i+1].strftime("%Y-%m-%d")
    mp = get_mp(df[start : end])
    dataset += [mp]

print(np.array(dataset).shape)

n_clusters=5

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
t0 = time()
kmeans.fit(dataset)
print("kmeans :\t%.2fs" % (time() - t0))
print(kmeans.labels_)

ward = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
t0 = time()
ward.fit(dataset)
print("%s :\t%.2fs" % ('ward', time() - t0))
print(ward.labels_)

average = AgglomerativeClustering(linkage='average', n_clusters=n_clusters)
t0 = time()
average.fit(dataset)
print("%s :\t%.2fs" % ('average', time() - t0))
print(average.labels_)

complete = AgglomerativeClustering(linkage='complete', n_clusters=n_clusters)
t0 = time()
complete.fit(dataset)
print("%s :\t%.2fs" % ('complete', time() - t0))
print(complete.labels_)

single = AgglomerativeClustering(linkage='single', n_clusters=n_clusters)
t0 = time()
single.fit(dataset)
print("%s :\t%.2fs" % ('single', time() - t0))
print(single.labels_)

labels = np.array([kmeans.labels_, ward.labels_, average.labels_, complete.labels_, single.labels_])

for cl in range(5):
    s=[]
    for k in [0,1,3,0]:
        l=list(np.where(labels[k,:]==cl)[0])
        s+=sample(l,k=5)

    fig, axs = plt.subplots(4, 5)
    fig.tight_layout()
    for i in range(20):
        axs[i//5, i%5].bar(np.arange(len(dataset[s[i]])),dataset[s[i]])
        axs[i//5, i%5].set_title(labels[:,s[i]])

    plt.show()
