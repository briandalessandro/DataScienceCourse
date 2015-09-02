import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def liftTable(pred, truth, b):
    df = pd.DataFrame({'p':pred + np.random.rand(len(pred))*0.000001, 'y':truth})
    df['b'] = b - pd.qcut(df['p'], b, labels=False)
    df['n'] = np.ones(df.shape[0])
    df_grp = df.groupby(['b']).sum()
    base = np.sum(df_grp['y'])/float(df.shape[0])
    df_grp['n_cum'] = np.cumsum(df_grp['n'])/float(df.shape[0])
    df_grp['y_cum'] = np.cumsum(df_grp['y'])
    df_grp['p_y_b'] = df_grp['y']/df_grp['n']
    df_grp['lift_b'] = df_grp['p_y_b']/base
    df_grp['cum_lift_b'] = (df_grp['y_cum']/(float(df.shape[0])*df_grp['n_cum']))/base
    return df_grp


def getMetrics(preds, labels):
    '''
    Takes in non-binary predictions and labels and returns AUC, and several Lifts
    '''
    auc = roc_auc_score(labels, preds)
    ltab = liftTable(preds, labels, 100)

    lift1 = ltab.ix[1].cum_lift_b
    lift5 = ltab.ix[5].cum_lift_b
    lift10 = ltab.ix[10].cum_lift_b
    lift25 = ltab.ix[25].cum_lift_b

    return [auc, lift1, lift5, lift10, lift25]


def dToString(d, dm1, dm2):
    '''
    Takes key-values and makes a string, d1 seprates k:v, d2 separates pairs
    '''
    arg_str = ''
    for k in sorted(d.keys()):
        if len(arg_str) == 0:
            arg_str = '{}{}{}'.format(k, dm1, d[k])
        else:
            arg_str = arg_str + '{}{}{}{}'.format(dm2, k, dm1, d[k])
    return arg_str

def getArgCombos(arg_lists):
    '''
    Takes every combination and returns an iterable of dicts
    '''
    keys = sorted(arg_lists.keys())
    #Initialize the final iterable
    tot = 1
    for k in keys:
        tot = tot * len(arg_lists[k])
    iter = []
    #Fill it with empty dicts
    for i in range(tot):
        iter.append({})
    #Now fill each dictionary    
    kpass = 1
    for k in keys:
        klist = arg_lists[k]
        ktot = len(klist)
        for i in range(tot):
            iter[i][k] = klist[(i/kpass) % ktot]
        kpass = ktot * kpass
    return iter


class LRAdaptor(object):
    '''
    This adapts the LogisticRegression() Classifier so that LR can be used as an init for GBT
    This just overwrites the predict method to be predict_proba
    '''
    def __init__(self, est):
        self.est = est

    def predict(self, X):
        return self.est.predict_proba(X)[:,1][:, np.newaxis]

    def fit(self, X, y):
        self.est.fit(X, y)

class GenericClassifier(object):

    def __init__(self, modclass, dictargs):
        self.classifier = modclass(**dictargs)

    def fit(self, X, Y):
        self.classifier.fit(X,Y)

    def predict_proba(self, Xt):
        return self.classifier.predict_proba(Xt)


class GenericClassifierOptimizer(object):

    def __init__(self, classtype, arg_lists):
        self.name = classtype.__name__
        self.classtype = classtype
        self.arg_lists = arg_lists
        self.results = self._initDict()

    def _initDict(self):
        return {'alg':[], 'opt':[], 'auc':[], 'lift1':[], 'lift5':[], 'lift10':[], 'lift25':[]}

    def _updateResDict(self, opt, perf):
        self.results['alg'].append(self.name)
        self.results['opt'].append(opt)
        self.results['auc'].append(perf[0])
        self.results['lift1'].append(perf[1])
        self.results['lift5'].append(perf[2])
        self.results['lift10'].append(perf[3])
        self.results['lift25'].append(perf[4])
   
    def runClassBake(self, X_train, Y_train, X_test, Y_test):

        arg_loop = getArgCombos(self.arg_lists)

        for d in arg_loop:
           
            mod = GenericClassifier(self.classtype, d)
            mod.fit(X_train, Y_train)
         
            perf = getMetrics(mod.predict_proba(X_test)[:,1], Y_test)
            self._updateResDict(dToString(d, ':', '|'), perf)



class ClassifierBakeoff(object): 

    def __init__(self, X_train, Y_train, X_test, Y_test, setup):
        self.instructions = setup
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.results = self._initDict()

    def _initDict(self):
        return {'alg':[], 'opt':[], 'auc':[], 'lift1':[], 'lift5':[], 'lift10':[], 'lift25':[]}

    def _updateResDict(self, clfr_results):
        self.results['alg'] =  self.results['alg'] + clfr_results['alg']
        self.results['opt'] =  self.results['opt'] + clfr_results['opt']
        self.results['auc'] =  self.results['auc'] + clfr_results['auc']
        self.results['lift1'] =  self.results['lift1'] + clfr_results['lift1']
        self.results['lift5'] =  self.results['lift5'] + clfr_results['lift5']
        self.results['lift10'] =  self.results['lift10'] + clfr_results['lift10']
        self.results['lift25'] =  self.results['lift25'] + clfr_results['lift25']


    def bake(self):

        for clfr in self.instructions:

           classifierBake = GenericClassifierOptimizer(clfr, self.instructions[clfr])
           classifierBake.runClassBake(self.X_train, self.Y_train, self.X_test, self.Y_test)
           self._updateResDict(classifierBake.results)






