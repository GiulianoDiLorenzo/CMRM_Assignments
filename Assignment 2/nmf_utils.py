import numpy as np


def template_pitch(K, pitch, freq_res, tol_pitch=0.05):
    """Obtain spectral template for a given pitch

    Args:
        K: Frequency bin(s)
        pitch: Given pitch
        freq_res: Frequency resolution
        tol_pitch: Accepted pitch tolerance

    Returns:
        template: Nonnegative vector of size K
    """
    
    max_freq = K * freq_res
    pitch_freq = 2**((pitch - 69) / 12) * 440
    max_order = int(np.ceil(max_freq / ((1 - tol_pitch) * pitch_freq)))
    template = np.zeros(K)
    for m in range(1, max_order + 1):
        min_idx = max(0, int((1- tol_pitch) * m * pitch_freq / freq_res))
        max_idx = min(K-1, int((1 + tol_pitch) * m * pitch_freq / freq_res))
        template[min_idx:max_idx+1] = 1/m

    return template


def initialize_template(K, pitch_set, freq_res, tol_pitch=0.05):
    """Initialize template matrix with onsets for a given set of pitches

    Args:
        K: Frequency bins
        pitch_set: Given set of pitches
        freq_res: Frequency resolution
        tol_pitch: Accepted pitch tolerance

    Returns:
        W: Nonnegative matrix of size K x (2R) with R = len(pitch_set)
    """
    
    R = len(pitch_set)
    W = np.zeros((K,2*R))
    for r in range(R):
        W[:,2*r] = 0.1
        W[:,2*r+1] = template_pitch(K, pitch_set[r], freq_res, tol_pitch=tol_pitch)

    return W


def initialize_activation(N, annotations, frame_res, tol_note=[0.2, 0.5], tol_onset=[0.3, 0.1], pitch_set=None):
    """Initialize activation matrix with onsets for given score annotations

    Args:
        N: Length in samples
        annotations: List of note annotations in the form [[start, duration, pitch, velocity, label]]
        frame_res: Frame resolutions
        tol_note: Accepted note tolerance
        tol_onset: Accepted onset tolerance
        pitch_set: Given set of pitches

    Returns:
        H: Nonnegative matrix of size (2R) x N with R = len(pitch_set)
        pitch_set: Pitch set
        label_pitch: Pitch labels
    """
    
    note_start = np.array([c[0] for c in annotations])
    note_dur = np.array([c[1] for c in annotations])
    pitch_all = np.array([c[2] for c in annotations])

    if pitch_set is None:
        pitch_set = np.unique(pitch_all)

    R = len(pitch_set)
    H = np.zeros((2*R,N))

    for i in range(len(note_start)):
        start_idx = max(0, int((note_start[i] - tol_note[0]) / frame_res))
        end_idx = min(N, int((note_start[i] + note_dur[i] + tol_note[1]) / frame_res))
        start_onset_idx = max(0, int((note_start[i] - tol_onset[0]) / frame_res))
        end_onset_idx = min(N, int((note_start[i] + tol_onset[1]) / frame_res))
        pitch_idx = np.argwhere(pitch_set == pitch_all[i])
        H[2*pitch_idx, start_onset_idx:end_onset_idx] = 1
        H[2*pitch_idx+1, start_idx:end_idx] = 1

    label_pitch = np.zeros(2*len(pitch_set),  dtype=int)

    for k in range(len(pitch_set)):
        label_pitch[2*k] = pitch_set[k]
        label_pitch[2*k+1] = pitch_set[k]
        
    return H, pitch_set, label_pitch