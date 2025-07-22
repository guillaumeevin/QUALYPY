###############
## Libraries ##
###############

import numpy as np

def clean_y(y):
    """
        return array where nan values have been replaced by zero values, and weights indicating
        where these values hqve been replaced    
    
		Parameters
		---------
		y: np.array[ shape = (n_samples) ]
			Raw time series
		"""

    # Identify NaN values
    nan_mask = np.isnan(y)

    # Create weights: 0 for NaN locations, 1 for valid data
    weights = np.ones_like(y, dtype=float)
    weights[nan_mask] = 0.

    # Replace NaN values in y (e.g., with 0)
    y_cleaned = np.copy(y)
    y_cleaned[nan_mask] = 0.
    
    return(y_cleaned, weights)