import glob
import pickle
import math

def calculate_bic(n, n_params, mse):
    """
    Calculate Bayesian Information Criterion (BIC).
    BIC = n * log(MSE) + log(n) * (parameters + 1) + n * log(2*pi) + n
    Using base-10 log.
    """
    if mse <= 0: return float('inf')
    return n * math.log10(mse) + math.log10(n) * (n_params + 1) + n * math.log10(2 * math.pi) + n

def load_all_mse_values():
    """Helper to load MSE values from all participant files."""
    rf_files = sorted(glob.glob("fit_results/*_model_3.pkl"))
    bs_files = sorted(glob.glob("fit_results/*_model_1.pkl"))
    
    rf_mses = []
    bs_mses = []

    for f in rf_files:
        with open(f, 'rb') as file: rf_mses.append(pickle.load(file)['fitResults'].fun)
            
    for f in bs_files:
        with open(f, 'rb') as file: bs_mses.append(pickle.load(file)['fitResults'].fun)
        
    return rf_mses, bs_mses

# --- Main Calculation ---
rf_mses, bs_mses = load_all_mse_values()
n = 120 # constant: 40 queries * 3 repetitions

# Initialize totals
total_bic = {"RF_6": 0, "RF_8": 0, "BS_7": 0, "BS_8": 0, "BS_9": 0, "BS_10": 0}

# Calculate RF BICs
for mse in rf_mses:
    total_bic["RF_6"] += calculate_bic(n, 6, mse) # a,b,c x 2
    total_bic["RF_8"] += calculate_bic(n, 8, mse) # a,b,c,d x 2

# Calculate BS BICs
for mse in bs_mses:
    total_bic["BS_7"]  += calculate_bic(n, 7,  mse) # a,b,c x 2 + 1 BS
    total_bic["BS_8"]  += calculate_bic(n, 8,  mse) # a,b,c x 2 + beta + N
    total_bic["BS_9"]  += calculate_bic(n, 9,  mse) # a,b,c,d x 2 + 1 BS
    total_bic["BS_10"] += calculate_bic(n, 10, mse) # a,b,c,d x 2 + beta + N

# --- Print Results ---
print("\nModel class - Free parameters - Total nr parameters - BIC")
print(f"RF - a,b,c x 2 - 6 - {total_bic['RF_6']:.5f}")
print(f"RF - a,b,c,d x 2 - 8 - {total_bic['RF_8']:.5f}")
print(f"BS - a,b,c x 2 + 1 BS - 7 - {total_bic['BS_7']:.5f}")
print(f"BS - a,b,c x 2 + beta + N - 8 - {total_bic['BS_8']:.5f}")
print(f"BS - a,b,c,d x 2 + 1 BS - 9 - {total_bic['BS_9']:.5f}")
print(f"BS - a,b,c,d x 2 + beta + N - 10 - {total_bic['BS_10']:.5f}")
