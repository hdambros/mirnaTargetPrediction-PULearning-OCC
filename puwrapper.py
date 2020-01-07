import numpy as np
from numpy import random
from sklearn import base

class PUWrapper(object):
    def __init__(self,trad_clf,n_fold=5):
        self._trad_clf=trad_clf
        self._n_fold=n_fold

    def fit(self,X,s):
        self._trad_clf.fit(X,s)

        Xp=X[s==1]
        n=len(Xp)
        
        cv_split=np.arange(n)*self._n_fold/n
        cv_index=np.floor(cv_split[random.permutation(n)])
        cs=np.zeros(self._n_fold)
        for k in range(self._n_fold):
            Xptr=Xp[cv_index==k]
            cs[k]=np.mean(self._trad_clf.predict_proba(Xptr)[:,1])
        self.c_=cs.mean()
        return self

    def predict_proba(self,X):
        proba=self._trad_clf.predict_proba(X)/self.c_
        proba[np.where( proba > 1 ) ] = 1
        return proba

    def predict(self,X):
        proba=self.predict_proba(X)[:,1]
        return proba>=(0.5)
