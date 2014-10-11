import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


def evenSplit(dat,fld):
    '''
    Evenly splits the data on a given binary field, returns a shuffled dataframe
    '''    
    pos=dat[(dat[fld]==1)]
    neg=dat[(dat[fld]==0)]
    neg_shuf=neg.reindex(np.random.permutation(neg.index))
    fin_temp=pos.append(neg_shuf[:pos.shape[0]],ignore_index=True)
    fin_temp=fin_temp.reindex(np.random.permutation(fin_temp.index))
    return fin_temp


def downSample(dat,fld,mult):
    '''
    Evenly splits the data on a given binary field, returns a shuffled dataframe
    '''
    pos=dat[(dat[fld]==1)]
    neg=dat[(dat[fld]==0)]
    neg_shuf=neg.reindex(np.random.permutation(neg.index))
    tot=min(pos.shape[0]*mult,neg.shape[0])
    fin_temp=pos.append(neg_shuf[:tot],ignore_index=True)
    fin_temp=fin_temp.reindex(np.random.permutation(fin_temp.index))
    return fin_temp


def scaleData(d):
    '''
    This function takes data and normalizes it to have the same scale (num-min)/(max-min)
    '''
    #Note, by creating df_scale like this we preserve the index
    df_scale=pd.DataFrame(d.iloc[:,1],columns=['temp'])
    for c in d.columns.values:
        df_scale[c]=(d[c]-d[c].min())/(d[c].max()-d[c].min())
    return df_scale.drop('temp',1)


def plot_dec_line(mn,mx,b0,b1,a,col,lab):
    '''
    This function plots a line in a 2 dim space
    '''
    x = np.random.uniform(mn,mx,100)
    dec_line = map(lambda x_i: -1*(x_i*b0/b1+a/b1),x)
    plt.plot(x,dec_line,col,label=lab)



def plotSVM(X, Y, my_svm):
    '''
    Plots the separating line along with SV's and margin lines
    Code here derived or taken from this example http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
    '''
    # get the separating hyperplane
    w = my_svm.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(X.iloc[:,0].min(), X.iloc[:,1].max())
    yy = a * xx - (my_svm.intercept_[0]) / w[1]
    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = my_svm.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = my_svm.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])
    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.scatter(my_svm.support_vectors_[:, 0], my_svm.support_vectors_[:, 1], s=80, facecolors='none')
    plt.plot(X[(Y==-1)].iloc[:,0], X[(Y==-1)].iloc[:,1],'r.')
    plt.plot(X[(Y==1)].iloc[:,0], X[(Y==1)].iloc[:,1],'b+')
    #plt.axis('tight')
    #plt.show()


