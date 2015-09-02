import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def getMAE(pred, truth):
    return np.abs(truth - pred).mean()

def getLL(pred, truth):
    ll_sum = 0
    for i in range(len(pred)):
        if (pred[i] == 0):
            p = 0.0001
        elif (pred[i] == 1):
            p = 0.9999
        else:
            p = pred[i]
        ll_sum += truth[i]*np.log(p)+(1-truth[i])*np.log(1-p)
    return (ll_sum)/len(pred)


def plotCalib(truth, pred, bins = 100, f = 0, l = '', w = 8, h = 8, fig_i = 1, fig_j = 1, fig_k = 1):
    mae = np.round(getMAE(pred, truth),3)
    ll = np.round(getLL(pred, truth), 3)

    d = pd.DataFrame({'p':pred, 'y':truth})
    d['p_bin'] = np.floor(d['p']*bins)/bins
    d_bin = d.groupby(['p_bin']).agg([np.mean, len])
    filt = (d_bin['p']['len']>f)


    if fig_k == 1:
        fig = plt.figure(facecolor = 'w', figsize = (w, h))

    x = d_bin['p']['mean'][filt]
    y = d_bin['y']['mean'][filt]
    n = d_bin['y']['len'][filt]

    stderr = np.sqrt(y * (1 - y)/n)

    ax = plt.subplot(fig_i, fig_j, fig_k)
    #plt.plot(x, y, 'b.', markersize = 9)
    plt.errorbar(x, y, yerr = 1.96 * stderr, fmt = 'o') 
    plt.plot([0.0, 1.0], [0.0, 1.0], 'k-')
    plt.title(l + ':' + ' MAE = {}, LL = {}'.format(mae, ll))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('prediction P(Y|X)')
    plt.ylabel('actual P(Y|X)')
    #plt.legend(loc=4)
    


def liftTable(pred, truth, b):
    df = pd.DataFrame({'p':pred + np.random.rand(len(pred))*0.000001, 'y':truth})
    df['b'] = b - pd.qcut(df['p'], b, labels=False)
    df['n'] = np.ones(df.shape[0])
    df_grp = df.groupby(['b']).sum()
    tot_y = float(np.sum(df_grp['y']))
    base = tot_y/float(df.shape[0])
    df_grp['n_cum'] = np.cumsum(df_grp['n'])/float(df.shape[0])
    df_grp['y_cum'] = np.cumsum(df_grp['y'])
    df_grp['p_y_b'] = df_grp['y']/df_grp['n']
    df_grp['lift_b'] = df_grp['p_y_b']/base
    df_grp['cum_lift_b'] = (df_grp['y_cum']/(float(df.shape[0])*df_grp['n_cum']))/base
    df_grp['recall'] = df_grp['y_cum']/tot_y
    return df_grp


def liftRecallCurve(pred, truth, b, h = 6, w = 12, title = ''):

    #Get the lift table
    lt = liftTable(pred, truth, b)

    fig, ax1 = plt.subplots(figsize = (w, h))

    ax1.plot(lt['n_cum'], lt['cum_lift_b'], 'b-')

    ax1.set_xlabel('Quantile')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('Lift', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    ax2.plot(lt['n_cum'], lt['recall'], 'r.')
    ax2.set_ylabel('Recall', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    plt.title(title)

    plt.show()

