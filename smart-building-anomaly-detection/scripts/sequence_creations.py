import numpy as np

def create_sequences(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Convert scaled 2D array into overlapping 3D windows for LSTM input.

    Parameters
    ----------
    data        : Scaled 2D array (n_samples, n_features)
    window_size : Number of timesteps per sequence

    Returns
    -------
    sequences : 3D array (n_windows, window_size, n_features)
    """
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i : i + window_size])
    return np.array(sequences)