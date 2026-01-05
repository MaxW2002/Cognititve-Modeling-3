def bayesFunction(p_h, p_d_given_h, p_d_given_not_h):
    """
    Calculates P(H|D) using Bayes' theorem for two competing hypotheses.
    
    Args:
        p_h: Prior probability of hypothesis H, P(H)
        p_d_given_h: Likelihood of data D given H, P(D|H)
        p_d_given_not_h: Likelihood of data D given not H, P(D|!H)
        
    Returns:
        Posterior probability of H given D, P(H|D)
    """
    # Calculate P(!H)
    p_not_h = 1 - p_h
    
    # Calculate P(D) using the law of total probability
    # P(D) = P(D|H)P(H) + P(D|!H)P(!H)
    p_d = (p_d_given_h * p_h) + (p_d_given_not_h * p_not_h)
    
    # Calculate P(H|D)
    p_h_given_d = (p_d_given_h * p_h) / p_d
    
    return p_h_given_d

def bayesFunctionMultipleHypotheses(priors, likelihoods):
    """
    Calculates P(H1|D) given vectors of priors and likelihoods for multiple hypotheses.
    
    Args:
        priors: List of prior probabilities for all hypotheses [P(H1), P(H2), ...]
        likelihoods: List of likelihoods of data given each hypothesis [P(D|H1), P(D|H2), ...]
        
    Returns:
        Posterior probability of the first hypothesis given D, P(H1|D)
    """
    # Calculate P(D) using the law of total probability
    # P(D) = sum(P(D|Hi) * P(Hi)) for all i
    p_d = sum(p * l for p, l in zip(priors, likelihoods))
    
    # Calculate P(H1|D)
    # The first item in each vector relates to the item of interest
    p_h1 = priors[0]
    p_d_given_h1 = likelihoods[0]
    
    p_h1_given_d = (p_d_given_h1 * p_h1) / p_d
    
    return p_h1_given_d

def bayesFactor(posteriors, priors):
    """
    Calculates and prints Bayes Factors.
    
    Args:
        posteriors: Vector of posterior probabilities [P(H1|D), P(H2|D), ...]
        priors: Vector of prior probabilities [P(H1), P(H2), ...]
        
    Returns:
        Dictionary containing the calculated Bayes Factors.
    """
    results = {}
    
    # BF 1 vs not 1
    # Posterior odds 1 vs not 1
    # Avoid division by zero if posterior is 1 (though unlikely in these examples)
    post_odds_1_not_1 = posteriors[0] / (1 - posteriors[0])
    prior_odds_1_not_1 = priors[0] / (1 - priors[0])
    
    bf_1_not_1 = post_odds_1_not_1 / prior_odds_1_not_1
    print(f'[1] "BF 1 vs not 1: {bf_1_not_1}"')
    results['BF_1_vs_not_1'] = bf_1_not_1
    
    # BF 1 vs others
    for i in range(1, len(posteriors)):
        # Posterior odds 1 vs i+1 (using 1-based indexing for display)
        post_odds_1_i = posteriors[0] / posteriors[i]
        prior_odds_1_i = priors[0] / priors[i]
        
        bf_1_i = post_odds_1_i / prior_odds_1_i
        print(f'[1] "BF 1 vs {i+1} : {bf_1_i}"')
        results[f'BF_1_vs_{i+1}'] = bf_1_i
        
    return results

def main():
    print("Self-test 1.2 Results (Two Hypotheses Function):")
    print("-" * 30)
    
    # Test cases from the assignment
    test_cases_1_2 = [
        {'label': 'A', 'p_h': 0.1, 'p_d_h': 0.9, 'p_d_not_h': 0.3, 'expected': 0.25},
        {'label': 'B', 'p_h': 0.9, 'p_d_h': 0.9, 'p_d_not_h': 0.3, 'expected': 0.96},
        {'label': 'C', 'p_h': 0.9, 'p_d_h': 0.3, 'p_d_not_h': 0.9, 'expected': 0.75}
    ]
    
    for case in test_cases_1_2:
        result = bayesFunction(case['p_h'], case['p_d_h'], case['p_d_not_h'])
        print(f"Case {case['label']}:")
        print(f"  Input: P(H)={case['p_h']}, P(D|H)={case['p_d_h']}, P(D|!H)={case['p_d_not_h']}")
        print(f"  Output: {result:.4f}")
        print(f"  Expected: {case['expected']}")
        print("-" * 30)

    print("\nSelf-check 1.4 Results (Multiple Hypotheses Function):")
    print("-" * 30)
    
    # Re-run 1.2 cases with new function
    print("Re-running 1.2 cases:")
    for case in test_cases_1_2:
        priors = [case['p_h'], 1 - case['p_h']]
        likelihoods = [case['p_d_h'], case['p_d_not_h']]
        result = bayesFunctionMultipleHypotheses(priors, likelihoods)
        print(f"Case {case['label']}:")
        print(f"  Input: Priors={priors}, Likelihoods={likelihoods}")
        print(f"  Output: {result:.4f}")
        print(f"  Expected: {case['expected']}")
        print("-" * 30)

    # New cases F, G, H
    test_cases_1_4 = [
        {'label': 'F', 'priors': [0.4, 0.3, 0.3], 'likelihoods': [0.99, 0.9, 0.2], 'expected': 0.545},
        {'label': 'G', 'priors': [0.4, 0.3, 0.3], 'likelihoods': [0.9, 0.9, 0.2], 'expected': 0.522},
        {'label': 'H', 'priors': [0.3, 0.3, 0.4], 'likelihoods': [0.9, 0.9, 0.2], 'expected': 0.435}
    ]

    print("Running 1.4 cases:")
    for case in test_cases_1_4:
        result = bayesFunctionMultipleHypotheses(case['priors'], case['likelihoods'])
        print(f"Case {case['label']}:")
        print(f"  Input: Priors={case['priors']}, Likelihoods={case['likelihoods']}")
        print(f"  Output: {result:.3f}")
        print(f"  Expected: {case['expected']}")
        print("-" * 30)

    print("\nSelf-check 1.6 Results (Bayes Factor):")
    print("-" * 30)
    
    # Case 1
    print("> bayesFactor(c(0.9,0.05,0.05), c(0.2,0.6,0.2))")
    bayesFactor([0.9, 0.05, 0.05], [0.2, 0.6, 0.2])
    print()
    
    # Case 2
    print("> bayesFactor(c(0.85,0.05,0.1),c(0.2,0.6,0.2))")
    bayesFactor([0.85, 0.05, 0.1], [0.2, 0.6, 0.2])
    print()
    
    # Case 3
    print("> bayesFactor(c(0.15,0.35,0.5),c(0.4,0.3,0.3))")
    bayesFactor([0.15, 0.35, 0.5], [0.4, 0.3, 0.3])

if __name__ == "__main__":
    main()
