import pickle
import sys
import scipy.optimize

try:
    with open('fit_results/part_0_model_3.pkl', 'rb') as f:
        d = pickle.load(f)
        print("Keys:", d.keys())
        print("MSE from fun:", d['fitResults'].fun)
except Exception as e:
    print(e)
