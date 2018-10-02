import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as skm
import warnings
warnings.filterwarnings('ignore')
from sklearn import linear_model

def simPolynomial(sigma = 0, betas = [0, 0], n = 100):

    x = np.random.uniform(0, 100, int(n))
    e = np.random.normal(0, sigma, int(n))

    d = pd.DataFrame(x, columns=['x'])    
    y = e
    for i, b in enumerate(betas):
        y = y + b*(x**i)
    d['y'] = y
    return d


def fitLinReg(d, mn, mx, inter):
    '''
    Runs a linear regression and fits it on a grid
    '''

    regr = linear_model.LinearRegression(fit_intercept = inter)
    regr.fit(d.drop('y', 1), d['y'])
    yhat = regr.predict(pd.DataFrame(np.arange(mn, mx, 1)))

    return yhat

def makePolyFeat(d, deg):
    '''
    Goal: Generate features up to X**deg
    1. a data frame with two features X and Y
    4. a degree 'deg' (from which we make polynomial features 
    
    '''
    #Generate Polynomial terms
    for i in range(2, deg+1):
        d['x'+str(i)] = d['x']**i
    return d

def fitFullReg(d, mn, mx, betas, inter):
    '''
    Runs a linear regression and fits it on a grid. Creates polynomial features using the dimension of betas
    '''

    regr = linear_model.LinearRegression(fit_intercept = inter)
    regr.fit(makePolyFeat(d.drop('y', 1), len(betas)), d['y'])
    dt = pd.DataFrame(np.arange(mn, mx, 1), columns = ['x'])
    yhat = regr.predict(makePolyFeat(dt, len(betas)))

    return yhat



def plotLinearBiasStage(sigma, betas, ns, fs):

    mn = 0
    mx = 101

    d = simPolynomial(sigma, betas, 10000)
    plt.figure(figsize = fs)
    plt.plot(d['x'], d['y'], 'b.', markersize = 0.75)


    x = np.arange(mn, mx, 1)
    y_real = np.zeros(len(x))
    for i, b in enumerate(betas):
        y_real += b*(x**i)

    #plt.plot(x, y_real + 2*sigma, 'k+')
    #plt.plot(x, y_real - 2*sigma, 'k--')
    plt.plot(x, y_real, 'k*')    

    for n in ns:
        dn = simPolynomial(sigma, betas, n)
        yhat = fitLinReg(dn, mn, mx, True)
        plt.plot(x, yhat, label = 'n={}'.format(n))


    plt.legend(loc = 4, ncol = 3)



def plotVariance(sigma, betas, ns, fs):

    mn = 0
    mx = 101
    nworlds = 100

    d = simPolynomial(sigma, betas, 10000)
    x = np.arange(mn, mx, 1)

    fig = plt.figure(figsize = fs)
    for pos, n in enumerate(ns):
       
        #First model each world
        yhat_lin = []
        yhat_non = []
        for i in range(nworlds):

            dn = simPolynomial(sigma, betas, n)

            yhat_lin.append(fitLinReg(dn, mn, mx, True))
            yhat_non.append(fitFullReg(dn, mn, mx, betas, True))

        #Now compute appropriate stats and plot

        lin_df = pd.DataFrame(yhat_lin)
        non_df = pd.DataFrame(yhat_non)

        lin_sig = lin_df.apply(np.std, axis=0).values
        non_sig = non_df.apply(np.std, axis=0).values
        lin_mu = lin_df.apply(np.mean, axis=0).values
        non_mu = non_df.apply(np.mean, axis=0).values

        #Need to continue from here

        for i in range(nworlds):
    
            ax1 = fig.add_subplot(2, 3, pos + 1)
            plt.title('n={}'.format(n))
            plt.plot(x, yhat_lin[i], '.', color = '0.75')
   
            if i == nworlds - 1:
                plt.plot(x, lin_mu, 'r-')
                plt.title('E[std|X] = {}'.format(round(lin_sig.mean(),1)))

            ax1.axes.get_xaxis().set_visible(False)
            ax1.set_ylim((-40, 80))

            ax2 = fig.add_subplot(2, 3, pos + 4)
            plt.plot(x, yhat_non[i], '--',  color = '0.75')

            if i == nworlds - 1:
                plt.plot(x, non_mu, 'r-')
                plt.title('E[std|X] = {}'.format(round(non_sig.mean(),1)))

            ax2.set_ylim((-40, 80)) 

            if pos != 0:
                ax1.axes.get_yaxis().set_visible(False)
                ax2.axes.get_yaxis().set_visible(False)

    plt.legend()







def getVarianceTrend(sigma, betas):

    mn = 50
    mx = 51
    nworlds = 100
    ns = np.logspace(4, 16, num = 10, base = 2)

    res_dict = {'n':[], 'lin':[], 'quad':[], 'non':[]}

    for pos, n in enumerate(ns):

        yhat_lin = []; yhat_quad = []; yhat_non = []

        for i in range(nworlds):

            dn = simPolynomial(sigma, betas, n)

            #yhat_lin.append(fitLinReg(dn, mn, mx, True)[0])
            yhat_lin.append(fitFullReg(dn, mn, mx, betas[0:1], True)[0])
            yhat_quad.append(fitFullReg(dn, mn, mx, betas[0:2], True)[0])
            yhat_non.append(fitFullReg(dn, mn, mx, betas, True)[0])

        res_dict['lin'].append(np.array(yhat_lin).std())
        res_dict['quad'].append(np.array(yhat_quad).std())
        res_dict['non'].append(np.array(yhat_non).std())
        res_dict['n'].append(n)


    return res_dict

def plotVarianceTrend(res_dict, fs):

    fig = plt.figure(figsize = fs)

    ax1 = fig.add_subplot(2, 1, 1)
    x = np.log2(res_dict['n'])
    plt.plot(x, np.power(res_dict['lin'], 2), 'b-', label = 'd = 1')
    plt.plot(x, np.power(res_dict['quad'], 2), 'r-', label = 'd = 2')
    plt.plot(x, np.power(res_dict['non'], 2), 'g-', label = 'd = 4')

    ax1.set_ylim((0, 100))

    plt.title('Model Variance by Polynomial Order (d) and Sample Size (n)')
    plt.legend(loc = 1)
    plt.ylabel('Var( E_d[Y|X = 50] )')

    ax2 = fig.add_subplot(2, 1, 2)
    filt = (x > 0)
    plt.plot(x[filt], 2*np.log2(res_dict['lin']), 'b-', label = 'd = 1')
    plt.plot(x[filt], 2*np.log2(res_dict['quad']), 'r-', label = 'd = 2')
    plt.plot(x[filt], 2*np.log2(res_dict['non']), 'g-', label = 'd = 4')

    ax2.set_xlim((x[filt].min(), x.max()))
    plt.xlabel('Log2(Sample Size)')
    plt.ylabel('Log [ Var( E_d[Y|X = 50] ) ]')  
    plt.legend(loc = 1)    
