import glob
import pickle
import os

def calculate_total_bic(model_suffix, model_name):
    # fit_results/part_X_model_Y.pkl
    pattern = f"fit_results/*_{model_suffix}.pkl"
    files = glob.glob(pattern)
    files.sort()
    
    total_bic = 0
    count = 0
    missing = []
    
    # Check for all 84 participants (0 to 83)
    # The filenames are part_0_..., part_1_... 
    # But wait, looking at file list, they are part_0, part_1... part_83. 
    # Let's just sum whatever we find, but also check if we have 84 unique participants.
    
    seen_participants = set()
    
    for fname in files:
        try:
            with open(fname, 'rb') as f:
                data = pickle.load(f)
                # data is dict with 'bic'
                if 'bic' in data:
                    total_bic += data['bic']
                    count += 1
                    
                    # Extract participant index
                    basename = os.path.basename(fname)
                    # format: part_X_model_Y.pkl
                    parts = basename.split('_')
                    if len(parts) >= 2 and parts[1].isdigit():
                        seen_participants.add(int(parts[1]))
                else:
                    print(f"Warning: 'bic' key not found in {fname}")
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            
    print(f"--- {model_name} (Suffix: {model_suffix}) ---")
    print(f"Files processed: {count}")
    print(f"Unique participants: {len(seen_participants)}")
    print(f"Total BIC: {total_bic:.5f}")
    
    if len(seen_participants) < 84:
        print(f"Warning: Only {len(seen_participants)}/84 participants processed.")

print("Calculating Total BIC from fit_results files...")
calculate_total_bic("model_3", "Relative Frequency (RF)")
calculate_total_bic("model_1", "Bayesian Sampling (BS)")
