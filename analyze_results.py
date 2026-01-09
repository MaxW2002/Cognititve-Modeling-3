import pickle

with open('fit_results/bs_fixed_N_beta_results.pkl', 'rb') as f:
    data = pickle.load(f)

print("=" * 70)
print("FIXED N AND BETA GRID RESULTS SUMMARY")
print("=" * 70)
print(f"\nNumber of participants run: {data['nparts']}")
print(f"N values tested: {data['n_list']}")
print(f"Beta values tested: {data['beta_list']}")
print(f"Total model combinations: {len(data['n_list'])} × {len(data['beta_list'])} = {len(data['n_list']) * len(data['beta_list'])}")

results = data['results']

print("\n" + "=" * 70)
print("MEAN MSE FOR EACH (N, BETA) COMBINATION:")
print("=" * 70)

# Create table view
print("\nN \\ β    ", end="")
for beta in data['beta_list']:
    print(f"{beta:>10.1f}", end="")
print()
print("-" * 70)

for n in data['n_list']:
    print(f"{n:>6}   ", end="")
    for beta in data['beta_list']:
        mse = results[(n, beta)]['mean_mse']
        print(f"{mse:>10.6f}", end="")
    print()

print("\n" + "=" * 70)
print("KEY OBSERVATIONS:")
print("=" * 70)

# Find best and worst
all_mses = [(n, b, results[(n, b)]['mean_mse']) for n in data['n_list'] for b in data['beta_list']]
all_mses_sorted = sorted(all_mses, key=lambda x: x[2])

print(f"\nBest fit: N={all_mses_sorted[0][0]}, beta={all_mses_sorted[0][1]} → MSE={all_mses_sorted[0][2]:.6f}")
print(f"Worst fit: N={all_mses_sorted[-1][0]}, beta={all_mses_sorted[-1][1]} → MSE={all_mses_sorted[-1][2]:.6f}")

# For each N, show min and max
print("\n" + "-" * 70)
print("For each N: minimum and maximum MSE (with corresponding beta):")
print("-" * 70)
for n in data['n_list']:
    betas_for_n = [(b, results[(n, b)]['mean_mse']) for b in data['beta_list']]
    best_b, best_mse = min(betas_for_n, key=lambda x: x[1])
    worst_b, worst_mse = max(betas_for_n, key=lambda x: x[1])
    print(f"N={n:>3}: Min MSE={best_mse:.6f} (β={best_b}), Max MSE={worst_mse:.6f} (β={worst_b})")

# For each beta, show trend across N
print("\n" + "-" * 70)
print("For each Beta: MSE trend across increasing N values:")
print("-" * 70)
for beta in data['beta_list']:
    n_mses = [(n, results[(n, beta)]['mean_mse']) for n in data['n_list']]
    print(f"β={beta:>3.1f}: ", end="")
    for n, mse in n_mses:
        print(f"N{n:>3}={mse:.4f}  ", end="")
    print()

print("\n" + "=" * 70)
print("INTERPRETATION:")
print("=" * 70)

# Compute average across betas for each N
print("\nAverage MSE across all beta values for each N:")
for n in data['n_list']:
    avg_mse = sum(results[(n, b)]['mean_mse'] for b in data['beta_list']) / len(data['beta_list'])
    print(f"  N={n:>3}: {avg_mse:.6f}")

# Compute average across N for each beta
print("\nAverage MSE across all N values for each beta:")
for beta in data['beta_list']:
    avg_mse = sum(results[(n, beta)]['mean_mse'] for n in data['n_list']) / len(data['n_list'])
    print(f"  β={beta:>3.1f}: {avg_mse:.6f}")

print("\n" + "=" * 70)
