import numpy as np
import sys
from numpy import shape
import time
from sklearn import neighbors, datasets, cross_validation, preprocessing, cluster
from sklearn.metrics import adjusted_rand_score
sys.path.append('./kshape/')
from kshape import kshape, zscore
from kshape import *
import os.path
import glob

for name in glob.glob('../data/*'):
    train_name=name[name.rfind('/')+1:]
    print train_name
    train_file=name+'/'+train_name+'_sparse_Train'
    test_file=name+'/'+train_name+'_sparse_Test'
    print train_name
    x_train_file=open(train_file,'r')
    x_test_file=open(test_file,'r')
    # Create empty lists
    x_total = []
    y_total=[]
    # Loop through datasets
    for x in x_train_file:
        temp=[float(ts) for ts in x.split(',')]
        x_total.append(temp[1:])
        y_total.append(int(temp[0]))
    for x in x_test_file:
        temp=[float(ts) for ts in x.split(',')]
        x_total.append(temp[1:])
        y_total.append(int(temp[0]))
    #preprocessing
    #relabeling
    print "Relabeling!"
    le=preprocessing.LabelEncoder()
    le.fit(y_total)
    y_total=le.transform(y_total)
    x_total=np.array(x_total)
    y_total=np.array(y_total)
    print "x_total.shape is ",x_total.shape
    n_cluster=np.max(y_total)+1
    print 'n_clusters=', n_cluster
    ###### begining kmeans clustering
    start=time.time()
    model=cluster.KMeans(n_clusters=n_cluster)#,init='random')
    model.fit(x_total)
    y=model.predict(x_total)
    from sklearn.metrics import normalized_mutual_info_score
    time_kmeans=time.time()-start
    nmi_kmeans=normalized_mutual_info_score(y_total,y)
    ri_kmeans=adjusted_rand_score(y_total,y)
    print 'time: %.3f, nmi: %.3f, ri: %.3f'%(time_kmeans,nmi_kmeans,ri_kmeans)

    ############# kshape part
    train_file=name+'/'+train_name;
    test_file=train_file+'_TEST'
    train_file=train_file+'_TRAIN'
    x_train_file=open(train_file,'r')
    x_test_file=open(test_file,'r')
    # Create empty lists
    x_total = []
    y_total=[]
    # Loop through datasets
    for x in x_train_file:
        temp=[float(ts) for ts in x.split(',')]
        x_total.append(temp[1:])
        y_total.append(int(temp[0]))
    for x in x_test_file:
        temp=[float(ts) for ts in x.split(',')]
        x_total.append(temp[1:])
        y_total.append(int(temp[0]))
    # normalize:
    #preprocessing
    #relabeling
    print "Relabeling!"
    le=preprocessing.LabelEncoder()
    le.fit(y_total)
    y_total=le.transform(y_total)
    x_total=np.array(x_total)
    y_total=np.array(y_total)
    print "x_total.shape is ",x_total.shape
    n_cluster=np.max(y_total)+1
    print 'n_clusters=', n_cluster
    start=time.time()
    clusters = kshape(zscore(x_total), n_cluster)
    #clusters=kshape(x_total,n_cluster)
    y_pred=np.zeros(x_total.shape[0])
    for i in range(len(clusters)):
        y_pred[clusters[i][1]]=i

    time_kshape=time.time()-start
    nmi_kshape=normalized_mutual_info_score(y_total,y_pred)
    ri_kshape=adjusted_rand_score(y_total,y_pred)
    print 'time: %.3f, nmi: %.3f, ri: %.3f'%(time_kshape,nmi_kshape,ri_kshape)

