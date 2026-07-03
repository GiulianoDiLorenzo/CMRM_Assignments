# Computer Music Representations and Models - Assignments

Course: Computer Music - Representations and Models

Programme: MSc in Music and Acoustic Engineering, Politecnico di Milano

## Overview

Python assignments, covering rhythm-based music genre classification using Support Vector Machines (SVMs) and score-informed source separation of vocal chorales via Nonnegative Matrix Factorization (NMF).

## HW1 - Music genre classification from rhythmic features

Classifies music tracks into 10 genres using **only rhythmic information** (no timbral or harmonic features), on the GTZAN dataset (1000 tracks, 10 genres, loaded via `deeplake`). Due to notebook memory constraints, trained on a 100-track subset with 20 held out for testing.

### Pipeline

1. **Preprocessing**: dynamic range normalization per track via `MinMaxScaler((-1, 1))`. Verified to materially affect downstream accuracy.
2. **Phase-based novelty function**: a custom onset-detection function built from scratch rather than a library call:
   - STFT of the signal, phase extracted and unwrapped via a `principal_argument` mapping (avoids phase-warping discontinuities).
   - Two successive discrete derivatives across time frames to isolate phase *acceleration*, each re-wrapped with `principal_argument`.
   - Accumulation across frequency bins, local-average subtraction (`compute_local_average`), half-wave rectification, and normalization.
   - Explored the effect of the three governing parameters: STFT window size `N`, hop size `H`, and local-average window `M` - on time/frequency resolution trade-offs and novelty curve smoothness.
3. **Feature vector construction**, per track, concatenates:
   - Phase novelty function (mean, std)
   - Tempogram (`librosa.feature.tempogram`)
   - Zero-crossing rate (`librosa.feature.zero_crossing_rate`): sensitive to time-domain envelope/percussive events
   - Spectral flux (`librosa.onset.onset_strength`): sensitive to frequency-domain/harmonic-rhythmic events
   - Tempo estimate (`librosa.beat.tempo`)

   Spectral flux is deliberately computed *before* the tempogram and reused as its `onset_envelope` argument: without this, the raw feature vector balloons past 10M samples per track and becomes unusable for training.
4. **Classification**: grid search over an SVM (`sklearn.svm.SVC`) across kernel type and `C`, with model caching (save/load via `joblib`) to avoid retraining.

### Results

Best-performing configuration was a linear kernel, reaching **100% training accuracy but only 15% test accuracy** - a clear overfitting signature: the `C` hyperparameter had no effect on the linear kernel's result, pointing to a feature vector with more redundant information than the ~100-track training set can constrain. The report ties this back concretely to preprocessing choices (unscaled audio reached 45% test accuracy vs. 15% scaled, since scaling flattens the dynamic cues rhythmic features depend on) and feature-computation resolution (subsampling the tempogram shifted test accuracy to 20%), rather than treating the SVM as a black box to tune blindly.

## HW2 - Score-informed source separation of vocal chorales

Separates the four canonical SATB voices (soprano, alto, tenor, bass) from a mixed chorale recording, using **NMF constrained by MIDI score information** rather than blind/unsupervised NMF. Applied to 4 chorales: two J.S. Bach settings (BWV 360, BWV 248-45) and two traditional Christmas songs ("Stille Nacht", "Joy to the World").

### Pipeline

1. **Annotation extraction**: `extract_annotations` parses per-voice MIDI files (`pretty_midi`) into `[start, duration, pitch, velocity, label]` tuples per note (238 notes for chorale 1), organized into a `pandas.DataFrame` and visualized as piano rolls per voice.
2. **Score-constrained NMF**: rather than random-initialized NMF, both factor matrices are built with explicit musical structure:
   - **Template matrix `W`** (`K × 2R`, `R` = number of distinct pitches): each column encodes a pitch's fundamental frequency plus a harmonic series with `1/m` weighting for the m-th harmonic, i.e. templates are handed as plausible harmonic spectrum rather than learned from scratch.
   - **Activation matrix `H`** (`2R × N`): initialized directly from the MIDI note timings (with configurable note/onset tolerances), so activations start out score-accurate rather than arbitrary.
   - `sklearn.decomposition.NMF` (multiplicative-update solver) then refines both matrices against the chorale's log-magnitude spectrogram, with `W`/`H` as custom initializations rather than random ones. Reconstruction error (2-norm between `V` and `WH`) reported per chorale.
3. **Hard-mask separation**: `split_annotations` partitions the note list by voice; the same score-informed activation-initialization is re-run once per voice to produce four "spectral masks," which are applied to `H` to isolate each voice's contribution before reconstructing its spectrogram as `W · H_voice`.
4. **Soft-mask refinement**: hard masks derived purely from MIDI timing ignore dynamics and produce audible artifacts, so a Wiener-style soft mask `M_voice = (W·H_voice) / (W·H + ε)` is computed per voice and applied directly to the original mixture STFT (preserving the true mixture phase) before inverse-STFT reconstruction.
5. **Batch pipeline + evaluation**: steps 1-4 wrapped into a single `separate()` function applied across all 4 chorales, evaluated two ways:
   - **Numerical**: 2-norm spectrogram error, plus SDR/SIR/SAR (via `mir_eval`), standard source-separation quality metrics in dB.
   - **Listening test**: informal perceptual rating per chorale/voice.

### Results

Separation quality varied substantially by chorale and voice - soprano consistently reconstructed cleanest, bass noisiest, attributed to higher pitches being better frequency-separated while lower voices' harmonics get misattributed across sources. Notably, **the numerical metrics and the listening test disagreed** on which chorales separated best (chorale 4 scored best on SDR/SIR/SAR but worse than chorales 1-2 by ear) - attributed to a timing-alignment mismatch between chorale and reference-source file lengths inflating the numerical error for chorales 3-4, a concrete example of why a single automatic metric shouldn't be trusted blindly in MIR evaluation.
