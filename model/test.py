import numpy as np
def generate_positive_integer_series_with_sum(n, m, mean=0, std_dev=1):
    # Step 1: Generate raw normal distribution
    X_raw = np.random.lognormal(mean, std_dev, m)
    
    # Step 2: Shift to ensure positivity
    shift = abs(min(X_raw)) + 1  # Ensure all values are positive
    X_shifted = X_raw + shift

    # Step 3: Scale to sum to n
    X_scaled = (n / np.sum(X_shifted)) * X_shifted

    # Step 4: Round to integers and ensure no zeros
    X_rounded = np.round(X_scaled).astype(int)
    X_rounded[X_rounded == 0] = 1  # Replace zeros with ones

    # Step 5: Adjust for sum to n
    delta = n - np.sum(X_rounded)
    while delta != 0:
        idx = np.random.choice(range(m))  # Randomly pick an index
        if delta > 0:
            X_rounded[idx] += 1
            delta -= 1
        elif delta < 0 and X_rounded[idx] > 1:
            X_rounded[idx] -= 1
            delta += 1

    return X_rounded
