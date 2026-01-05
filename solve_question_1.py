import sys
import os

# Add current directory to path to import assignment 3
sys.path.append(os.getcwd())
try:
    # Import functions from assignment 3.py
    # Note: The file name has a space, so we use importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("assignment_3", "assignment 3.py")
    assignment_3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assignment_3)
    
    bayesFunction = assignment_3.bayesFunction
    bayesFactor = assignment_3.bayesFactor
    bayesFunctionMultipleHypotheses = assignment_3.bayesFunctionMultipleHypotheses
except ImportError as e:
    print(f"Error importing assignment 3: {e}")
    sys.exit(1)

def solve_question_1():
    print("--- Question 1A ---")
    # A. Calculate posterior
    # P(H) = 0.5
    # P(D|H) = 0.531
    # P(D|!H) = 0.52
    p_h = 0.5
    p_d_h = 0.531
    p_d_not_h = 0.52
    
    # Use bayesFunction from assignment 3
    posterior_a = bayesFunction(p_h, p_d_h, p_d_not_h)
    print(f"Posterior P(H|D): {posterior_a}")

    print("\n--- Question 1B ---")
    # B. Bayes Factor
    # Use bayesFactor from assignment 3
    # We need posteriors and priors vectors
    # H1 = People can see future, H2 = People cannot see future
    # Priors: [0.5, 0.5]
    # Posteriors: [posterior_a, 1 - posterior_a]
    
    priors_b = [0.5, 0.5]
    posteriors_b = [posterior_a, 1 - posterior_a]
    
    print("Calculating Bayes Factor using assignment 3 function:")
    bf_results = bayesFactor(posteriors_b, priors_b)
    bf_val = bf_results['BF_1_vs_not_1']
    print(f"Bayes Factor (H vs !H): {bf_val}")

    print("\n--- Question 1C ---")
    # C. Skeptic researcher
    # Prior P(H) = 0.001
    p_h_skeptic = 0.001
    p_not_h_skeptic = 1 - p_h_skeptic
    
    # Calculate posterior odds using BF from 1B
    # Posterior Odds = BF * Prior Odds
    prior_odds_skeptic = p_h_skeptic / p_not_h_skeptic
    posterior_odds_skeptic = bf_val * prior_odds_skeptic
    
    print(f"Prior Odds (Skeptic): {prior_odds_skeptic}")
    print(f"Posterior Odds (Skeptic): {posterior_odds_skeptic}")
    
    # Verify with bayesFunction
    posterior_skeptic = bayesFunction(p_h_skeptic, p_d_h, p_d_not_h)
    print(f"Posterior Probability (Skeptic) calculated directly: {posterior_skeptic}")
    # Convert probability to odds to check
    posterior_odds_check = posterior_skeptic / (1 - posterior_skeptic)
    print(f"Posterior Odds (Check): {posterior_odds_check}")

    print("\n--- Question 1D ---")
    # D. Replicated experiments
    # "The priors are updated based on the outcomes of earlier experiments, starting with the outcome of the initial experiment."
    
    current_prior = posterior_a
    print(f"Initial Prior (from Exp 1): {current_prior}")

    experiments = [
        {'id': 2, 'p_d_h': 0.471, 'p_d_not_h': 0.520},
        {'id': 3, 'p_d_h': 0.491, 'p_d_not_h': 0.65},
        {'id': 4, 'p_d_h': 0.505, 'p_d_not_h': 0.70}
    ]

    cumulative_bf = bf_val # Start with BF from Exp 1

    for exp in experiments:
        # Use bayesFunction for update
        new_posterior = bayesFunction(current_prior, exp['p_d_h'], exp['p_d_not_h'])
        print(f"Experiment {exp['id']}:")
        print(f"  Prior: {current_prior}")
        print(f"  Likelihoods: P(D|H)={exp['p_d_h']}, P(D|!H)={exp['p_d_not_h']}")
        print(f"  Posterior: {new_posterior}")
        
        # Calculate BF for this step using bayesFactor function
        # We need temporary priors (flat 0.5 to get just the likelihood ratio as BF) or use the formula directly
        # Actually, bayesFactor function calculates BF based on Posterior/Prior odds.
        # So if we pass the current prior and new posterior, it should give us the BF for this step.
        step_priors = [current_prior, 1 - current_prior]
        step_posteriors = [new_posterior, 1 - new_posterior]
        
        print(f"  Calculating Step BF...")
        step_bf_results = bayesFactor(step_posteriors, step_priors)
        step_bf = step_bf_results['BF_1_vs_not_1']
        
        cumulative_bf *= step_bf
        
        # Update prior for next step
        current_prior = new_posterior

    print("\n--- Question 1E ---")
    # E. Argument using Bayes Factor
    print(f"Total Cumulative Bayes Factor: {cumulative_bf}")

if __name__ == "__main__":
    solve_question_1()
