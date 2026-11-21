import math

def detect_ddm_drift(error_stream, min_instances=30):
    """
    Drift Detection Method (DDM) by Gama et al. (2004).
    Takes a list of errors (1 for incorrect prediction, 0 for correct).
    Returns the index where drift is detected, or -1 if no drift is found.
    """
    p_min = float('inf')
    s_min = float('inf')
    error_count = 0
    
    for i, error in enumerate(error_stream):
        n = i + 1
        error_count += error
        
        # Calculate current probability of error and standard deviation
        p_i = error_count / n
        s_i = math.sqrt((p_i * (1 - p_i)) / n)
        
        # Wait for enough samples to establish a reliable baseline
        if n < min_instances:
            continue
            
        # Update the minimums if a new low is reached
        if p_i + s_i < p_min + s_min:
            p_min = p_i
            s_min = s_i
            
        # Check for drift (Warning level is 2*s_min, Drift level is 3*s_min)
        if p_i + s_i > p_min + 3 * s_min:
            return i  # Drift detected, trigger retraining
            
    return -1