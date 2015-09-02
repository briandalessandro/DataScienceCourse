'''
This script has a set of reference functions for performing analysis of the churn dataset
'''
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as skm
sys.path.append("./utils/")
from ClassifierBakeoff import *

import warnings
warnings.filterwarnings('ignore')

def getDfSummary(dat):
    '''
    Get descriptive stats
    '''
    #Get the names of the columns
    cols = dat.columns.values

    c_summ = []
    #Outer Loop for the cols
    for c in cols:
        #Count the NAs
        missing = sum(pd.isnull(dat[c]))
        #Use describe to get summary statistics, and also drop the 'count' row
        sumval = dat[c].describe().drop(['count'])
        #Now count distinct values...note that nunique removes missing values for you
        distinct = dat[c].nunique()
        #Append missing and distinct to sumval
        sumval = sumval.append(pd.Series([missing, distinct], index=['missing', 'distinct']))
        #Add each sumval to a list and then convert the entire thing to a DS
        c_summ.append(sumval)

    return pd.DataFrame(c_summ, index=cols)





def plotCorr(dat, lab, h, w):
    '''
    Do a heatmap to visualize the correlation matrix, dropping the label
    '''

    dat = dat.drop(lab, 1)
    #Get correlation and 0 out the diagonal (for plotting purposes)
    c_dat = dat.corr()
    for i in range(c_dat.shape[0]):
        c_dat.iloc[i,i] = 0

    c_mat = c_dat.as_matrix()
    #c_mat = c_mat[:-1, :-1]
    fig, ax = plt.subplots()
    heatmap = plt.pcolor(c_mat, cmap = plt.cm.RdBu)
 
    #Set the tick labels and center them
    ax.set_xticks(np.arange(c_dat.shape[0]) + 0.5, minor = False)
    ax.set_yticks(np.arange(c_dat.shape[1]) + 0.5, minor = False)
    ax.set_xticklabels(c_dat.index.values, minor = False, rotation = 45)
    ax.set_yticklabels(c_dat.index.values, minor = False)
    heatmap.axes.set_ylim(0, len(c_dat.index))  
    heatmap.axes.set_xlim(0, len(c_dat.index)) 
    plt.colorbar(heatmap, ax = ax)

    #plt.figure(figsize = (h, w))
    fig = plt.gcf()
    fig.set_size_inches(h, w)


def makeBar(df, h, lab,  width):
    '''
    Contains
    '''
    df_s = df.sort(columns = [h], ascending = False)

    #Get a barplot
    ind = np.arange(df_s.shape[0])
    labs = df_s[[lab]].values.ravel() 

    fig = plt.figure(facecolor = 'w', figsize = (12, 6))
    ax = plt.subplot(111)
    plt.subplots_adjust(bottom = 0.25)

    rec = ax.bar(ind + width, df_s[[h]].values, width, color='r')

    ax.set_xticks(ind + getTickAdj(labs, width))
    ax.set_xticklabels(labs, rotation = 45, size = 14)


def getTickAdj(labs, width):
    lens = map(len, labs)
    lens = -1 * width * (lens - np.mean(lens)) / np.max(lens)
    return lens

def plotMI(dat, lab, width = 0.35, signed = 0):
    '''
    Draw a bar chart of the normalized MI between each X and Y
    '''
    X = dat.drop(lab, 1)
    Y = dat[[lab]].values
    cols = X.columns.values
    mis = []

    #Start by getting MI
    for c in cols:
        mis.append(skm.normalized_mutual_info_score(Y.ravel(), X[[c]].values.ravel()))

    #Get signs by correlation
    corrs = dat.corr()[lab]
    corrs[corrs.index != lab]
    df = pd.DataFrame(zip(mis, cols), columns = ['MI', 'Lab'])
    df = pd.merge(df, pd.DataFrame(corrs, columns = ['corr']), how = 'inner', left_on = 'Lab', right_index=True)
 
    if signed == 0:
        makeBar(df, 'MI', 'Lab', width)

    else:
        makeBarSigned(df, 'MI', 'Lab', width)


def makeBarSigned(df, h, lab,  width):
    '''
    Contains
    '''
    df_s = df.sort(columns = [h], ascending = False)

    #Get a barplot
    ind = np.arange(df_s.shape[0])
    labs = df_s[[lab]].values.ravel()
    h_pos = (df_s[['corr']].values.ravel() > 0) * df_s.MI
    h_neg = (df_s[['corr']].values.ravel() < 0) * df_s.MI

    fig = plt.figure(facecolor = 'w', figsize = (12, 6))
    ax = plt.subplot(111)
    plt.subplots_adjust(bottom = 0.25)

    rec = ax.bar(ind + width, h_pos, width, color='r', label = 'Positive')
    rec = ax.bar(ind + width, h_neg, width, color='b', label = 'Negative')

    ax.set_xticks(ind + getTickAdj(labs, width))
    ax.set_xticklabels(labs, rotation = 45, size = 14)

    plt.legend()



def makeGS_Tup(ent, getmin = True):

    ostr = dToString(ent.parameters, ':', '|')
    if len(ostr.split('|')) > 2:
        sp = ostr.split('|')
        if len(sp) == 3:
            ostr = '{}|{}\n{}'.format(sp[0], sp[1], sp[2])
        else:
            ostr = '{}|{}\n{}|{}'.format(sp[0], sp[1], sp[2], sp[3])
        
    #ostr = dToString(ent.parameters, ':', '|')
    mu = np.abs(ent.mean_validation_score) #Log-Loss comes in at negative value
    sig = ent.cv_validation_scores.std()
    stderr = sig/np.sqrt(len(ent.cv_validation_scores))
                         
    if getmin:
        return (mu, ostr, mu + stderr, sig, stderr) #Note, this assumes minimization, thus adding stderr
    else:
        return (mu, ostr, mu - stderr, sig, stderr)
        
    
def rankGS_Params(gs_obj_list, getmin = True):
    '''
    Takes in the .grid_scores_ attributes of a GridSearchCV object
    '''
    tup_list = []
    
    for k in gs_obj_list:
        tup_list.append(makeGS_Tup(k, getmin))
    
    tup_list.sort()

    if not getmin:
        tup_list.reverse()

    return tup_list



def processGsObjList(gs_obj_list, getmin = True):

    rank_list = rankGS_Params(gs_obj_list, getmin)
    hts = []
    desc = []
    errs = []
    std1 = rank_list[0][4]

    for tup in rank_list:
        hts.append(tup[0])
        desc.append(tup[1])
        errs.append(2 * tup[4])

    return [hts, desc, errs, std1]

def plotGridSearchSingle(gs_obj_list, getmin = True):

    hts, desc, errs, std1 = processGsObjList(gs_obj_list, getmin = True)

    gridBarH(hts, desc, errs, std1)



def plotGridSearchMulti(tup_list, getmin = True):
    '''
    Loop through a list of gs_obj_lists. The Obj list is in the 1 slot of each value in the dict
    '''
    m_ht = []
    m_desc = []
    m_errs = []

    best_min = 1000 #This assumes we are minimizing

    for tup in tup_list:
        lab = tup[0]
        gs_dict = tup[1]

        for k in gs_dict:
            clf = type(k).__name__.split('Classifier')[0]

            hts, desc, errs, std1 = processGsObjList(gs_dict[k][1], getmin = True)
            for i, d in enumerate(desc):
                desc[i] = '{} {} {}'.format(clf, lab, d)

            if hts[0] < best_min:
                best_std1 = std1
    
            m_ht = m_ht + hts
            m_desc = m_desc + desc
            m_errs = m_errs + errs

    gridBarH(m_ht, m_desc, m_errs, best_std1, int(len(m_ht)), 12)



def gridBarH(hts, desc, errs, std1, h = 6, w = 12):

    fig = plt.figure(facecolor = 'w', figsize = (w, h))
    ax = plt.subplot(111)
    plt.subplots_adjust(bottom = 0.25)

    width = 0.5
    
    pos = np.arange(len(hts))

    rec = ax.barh(pos, np.array(hts), width, yerr = np.array(errs), color='r')

    ax.set_yticks(pos + width/2)
    ax.set_yticklabels(desc, size = 14)

    tmp = list(hts)
    tmp.sort()

    x_min = np.array(hts).min() - 2*np.array(hts).std()
    x_max = tmp[-2] + 2*np.array(hts).std()
    plt.xlim(x_min, x_max)


    plt.plot(tmp[0] * np.ones(len(tmp)), pos)
    plt.plot((tmp[0] + std1) * np.ones(len(tmp)), pos)












