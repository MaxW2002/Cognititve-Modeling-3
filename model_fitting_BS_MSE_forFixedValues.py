"""
Run the Bayesian Sampling model for fixed values of N and beta.
This script is a modified copy of `model_fitting_BS_MSE.py` and iterates
over specified N and beta values, fits only the free parameters a,b,c,d,
collects per-participant MSEs, computes the mean across participants for
each (N,beta) combination, and saves a plot of beta vs MSE for different N.

Default: runs first 10 participants. Use `--nparts` to change.
"""

import os
import numpy as np
import scipy.stats as st
from scipy.optimize import fmin, differential_evolution
import glob
import pandas as pd
import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def clean_data():
    all_data = glob.glob('all_data/*.csv')
    numPar = len(all_data)
    print(numPar,' participants were considered!')
    est = np.zeros(shape=(numPar,60,3))
    neg, land, lor, lg = ' not', ' and', ' or', ' given'
    eventAs = [' cold', ' windy', ' warm']
    eventBs = [' rainy', ' cloudy', ' snowy']
    queryOrder = []
    for A, B in zip(eventAs, eventBs):
        queryOrder.append(A)
        queryOrder.append(B)
        queryOrder.append(neg + A)
        queryOrder.append(neg + B)
        queryOrder.append(A + land + B)
        queryOrder.append(B + land + neg + A)
        queryOrder.append(A + land + neg + B)
        queryOrder.append(neg + A + land + neg + B)
        queryOrder.append(A + lor + B)
        queryOrder.append(B + lor + neg + A)
        queryOrder.append(A + lor + neg + B)
        queryOrder.append(neg + A + lor + neg + B)
        queryOrder.append(A + lg + B)
        queryOrder.append(neg + A + lg + B)
        queryOrder.append(A + lg + neg + B)
        queryOrder.append(neg + A + lg + neg + B)
        queryOrder.append(B + lg + A)
        queryOrder.append(neg + B + lg + A)
        queryOrder.append(B + lg + neg + A)
        queryOrder.append(neg + B + lg + neg + A)

    for i,fname in enumerate(all_data):
        print('Processing Participant No.%d' % (i+1), fname)
        df = pd.read_csv(fname)
        for j,q in enumerate(queryOrder):
            nowEst = df[df['querydetail']==q]['estimate']
            nowEstValues = nowEst.values/100
            for k in range(3):
                est[i,j,k] = nowEstValues[k]

    with open('pEstData.pkl','wb') as f:
        pickle.dump({'data': est, 'query_order': queryOrder}, f)
    return est, queryOrder


def get_truePr_BS(a,b,c,d):
    truePr = []
    base = a+b+c+d
    truePr.append((a+c)/base)
    truePr.append((a+b)/base)
    truePr.append((b+d)/base)
    truePr.append((c+d)/base)
    truePr.append(a/base)
    truePr.append(b/base)
    truePr.append(c/base)
    truePr.append(d/base)
    truePr.append((a+b+c)/base)
    truePr.append((a+b+d)/base)
    truePr.append((a+c+d)/base)
    truePr.append((b+c+d)/base)
    truePr.append((a/(a+b)))
    truePr.append((b/(a+b)))
    truePr.append((c/(c+d)))
    truePr.append((d/(c+d)))
    truePr.append((a/(a+c)))
    truePr.append((c/(a+c)))
    truePr.append((b/(b+d)))
    truePr.append((d/(b+d)))
    return truePr


# Globals that will be set for each exploration run
GLOBAL_BETA = None
GLOBAL_N = None
GLOBAL_TESTDATA = None


def generativeModel_BS_fixed(params):
    """params: eight free parameters [a0,b0,c0,d0,a1,b1,c1,d1]
    Uses GLOBAL_BETA and GLOBAL_N for the Bayesian components."""
    a,b,c,d = [0,0],[0,0],[0,0],[0,0]
    a[0],b[0],c[0],d[0],a[1],b[1],c[1],d[1] = params
    beta = GLOBAL_BETA
    N = GLOBAL_N
    N2 = N
    MSE = 0
    allpredmeans = np.zeros((40,))
    for iter in range(2):
        sum_of_truePr = a[iter]+b[iter]+c[iter]+d[iter]
        MSE += (sum_of_truePr/100-1)**2/2
        truePr = get_truePr_BS(a[iter], b[iter], c[iter], d[iter])
        for i, trueP in enumerate(truePr):
            if i<4 or i>=12:
                allpredmeans[i+iter*20] = trueP*N/(N+2*beta)+beta/(N+2*beta)
            else:
                allpredmeans[i+iter*20] = trueP*N2/(N2+2*beta)+beta/(N2+2*beta)
    return allpredmeans, MSE


def MSE_BS_fixed(params):
    allpredmeans, MSE = generativeModel_BS_fixed(params)
    testdata = GLOBAL_TESTDATA
    for i in range(len(allpredmeans)):
        currentdata = testdata[i,:].flatten()
        MSE += np.mean((allpredmeans[i] - currentdata) ** 2)/40
    return MSE


def run_fixed_grid(nparts=10, n_list=None, beta_list=None, outdir='fit_results'):
    if n_list is None:
        n_list = [1,2,5,10,50,100]
    if beta_list is None:
        beta_list = [0.1,0.5,1,2,5]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    global pData, queryOrder, GLOBAL_BETA, GLOBAL_N, GLOBAL_TESTDATA
    pData, queryOrder = clean_data()

    # limit participants
    nparticipants = min(nparts, pData.shape[0])
    print(f'\n***** Running analysis with {nparticipants} participants *****\n')

    results = {}

    # bounds for eight parameters (a,b,c,d for two scenarios)
    bnds = [(0.0, 100)] * 8

    for N in n_list:
        GLOBAL_N = N
        for beta in beta_list:
            GLOBAL_BETA = beta
            print('\nRunning for N=%s, beta=%s' % (N, beta))
            participant_mses = []
            participant_params = []
            for ipar in range(nparticipants):
                GLOBAL_TESTDATA = pData[ipar,:,:]
                fit_res = differential_evolution(MSE_BS_fixed, bounds=bnds,
                                                  popsize=30,
                                                  disp=False, polish=fmin, tol=1e-5)
                minMSE = fit_res.fun
                participant_mses.append(minMSE)
                participant_params.append(fit_res.x)
                print('  participant %d (of %d): MSE = %.6f' % (ipar+1, nparticipants, minMSE))

            mean_mse = float(np.mean(participant_mses))
            print('  >>> Mean MSE across all %d participants: %.6f' % (nparticipants, mean_mse))
            results[(N,beta)] = {'mean_mse': mean_mse,
                                  'participant_mses': participant_mses,
                                  'participant_params': participant_params}

    # save results
    savepath = os.path.join(outdir, 'bs_fixed_N_beta_results.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump({'results': results, 'nparts': nparticipants, 'n_list': n_list, 'beta_list': beta_list}, f)
    print('\nSaved results to', savepath)

    # plot: for each N plot beta vs mean_mse
    plt.figure(figsize=(10, 6))
    for N in n_list:
        ys = [results[(N,b)]['mean_mse'] for b in beta_list]
        plt.plot(beta_list, ys, marker='o', label='N=%s' % N)
    plt.xlabel('Beta')
    plt.ylabel('Mean MSE (averaged across %d participants)' % nparticipants)
    plt.title('BS model: Mean MSE across participants (varying Beta)')
    plt.legend()
    plt.grid(True)
    figpath = os.path.join(outdir, 'bs_fixed_N_beta_MSE.png')
    plt.savefig(figpath, dpi=200)
    print('Saved plot to', figpath)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nparts', type=int, default=10, help='number of participants to run (default 10)')
    parser.add_argument('--outdir', type=str, default='fit_results', help='output directory')
    args = parser.parse_args()
    run_fixed_grid(nparts=args.nparts, outdir=args.outdir)
