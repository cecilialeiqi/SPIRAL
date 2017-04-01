import numpy as np
#from knn_euclid_class import *
import sys
from numpy import shape
from sklearn import neighbors, datasets, cross_validation, preprocessing
import glob

for name in glob.glob('../data/*'):
    train_name=name[name.rfind('/')+1:]
    print train_name
    train_file=name+'/'+train_name+'_sparse_Train'
    test_file=name+'/'+train_name+'_sparse_Test'

    x_train_file=open(train_file,'r')
    x_test_file=open(test_file,'r')

    # Create empty lists
    Train=np.genfromtxt(train_file,delimiter=',')
    Test=np.genfromtxt(test_file,delimiter=',')
    x_train=Train[:,1:]
    y_train=Train[:,0]
    x_test=Test[:,1:]
    y_test=Test[:,0]

    x_total=np.vstack([x_train,x_test])
    y_total=np.hstack([y_train,y_test])

    #preprocessing
    #relabeling
    print "Relabeling!"
    from sklearn.preprocessing import normalize
    x_total=normalize(x_total,'l2')
    x_train=x_total[:len(x_train)]
    x_test=x_total[len(x_train):]

    le=preprocessing.LabelEncoder()
    le.fit(y_total)
    y_total=le.transform(y_total)
    #y_total=np.array(y_total)
    n_cluster=np.max(y_total)+1
    print 'n_clusters=', n_cluster

    p=x_train.shape[1]
    while (x_train[0,p-1]==0):
        p-=1
    x_train=x_train[:,:p]
    x_test=x_test[:,:p]
    print "x_train.shape is ",x_train.shape
    print "x_test.shape is ",x_test.shape

    print "Start training:"
    import time

    start=time.time()

    from sklearn.metrics import roc_auc_score

    lb=preprocessing.LabelBinarizer()
    lb.fit(y_train)

    def cross_val(X,y, model, N):
        mean_auc=0.
        l=np.random.permutation(X.shape[0])
        X=X[l,:]
        y=y[l]
        for i in range(N):
            X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                    X, y, test_size=1.0/float(N))
            try:
                model.fit(X_train, y_train)
            except:
                print "Error during training"
                continue;
            preds = model.predict_proba(X_cv)
            lb.fit(y_train)

            if n_cluster==2:
                auc=roc_auc_score(y_cv,preds[:,1])
            else:
                y_trans=lb.transform(y_cv)
                a=[]
                preds_=[]
                for j in range(y_trans.shape[1]):
                    if (len(np.unique(y_trans[:,j]))==1):
                        # unluckily in the CV set, some label has never appeared. Then we could just omit this label
                        continue
                    else:
                        a.append(y_trans[:,j])
                        preds_.append(preds[:,j])
                a=np.array(a).T
                preds_=np.array(preds_).T
                auc = roc_auc_score(a, preds_)
            mean_auc += auc
        return mean_auc/N

    #1) knn

    clf = neighbors.KNeighborsClassifier(n_neighbors=1, weights='uniform')
    clf.fit(x_train, y_train)
    label=clf.predict_proba(x_test)
    from sklearn.metrics import roc_auc_score

    time_knn=time.time()-start
    print "elapsed time of knn: ",time_knn
    if n_cluster==2:
        auc_knn=roc_auc_score(y_test,label[:,1])
    else:
        auc_knn=roc_auc_score(lb.transform(y_test),label)
    print 'AUC of KNN:', auc_knn

    # 2) use logistic regression
    bestscore=0
    Cvals = np.logspace(-7, 7, 50, base=1.5)
    bestC=0.1
    from sklearn.cross_validation import train_test_split

    from sklearn.linear_model import LogisticRegression
    for c in Cvals:
        if n_cluster==2:
            model = LogisticRegression(penalty='l2',C=c,class_weight='balanced')
        else:
            model = LogisticRegression(penalty='l2',C=c,class_weight='balanced', multi_class='ovr')
        score=cross_val(x_train,y_train,model,5)
        #print score
        if score>bestscore:
            bestC=c
            bestscore=score

    model.C=bestC
    model.fit(x_train,y_train)
    preds=model.predict_proba(x_test)
    lb.fit(y_train)
    a=lb.transform(y_test)
    if (a.shape[1]!=preds.shape[1]):
        a=np.vstack([a.T,np.ones(preds.shape[0])]).T
    if n_cluster==2:
        auc_lr=roc_auc_score(y_test,preds[:,1])
    else:
        auc_lr=roc_auc_score(a,preds)#lb.transform(y_test))
    print 'best C:', bestC,'-- AUC of logistic regression is ',auc_lr
    time_lr=time.time()-start
    print 'time elapsed:', time_lr

