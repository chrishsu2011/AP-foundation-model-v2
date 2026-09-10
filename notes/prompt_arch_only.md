## Task

Design a transformer architecture that generates Hodgkin-Huxley (HH) action potential waveforms conditioned on biophysical parameters. Do not train it yet — just build the model and data loading.

## Background

The dataset (`hh_dataset.h5`) contains 1000 single-compartment HH neuron simulations. Each trace is a voltage waveform (1000 timesteps, dt = 0.01 ms, 50 ms total) produced by a 1 ms current pulse at t = 10 ms. The neurons share the same HH equations but differ in 26 jittered biophysical parameters (conductances, reversal potentials, capacitance, stimulus amplitude, and rate-function kinetics). All parameters are stored alongside each waveform in the HDF5 file under `voltage` (shape 1000×1000), `parameters/` (26 arrays of length 1000), `stimulus/amplitude` (length 1000), and `time` (length 1000).

The long-term goal is to train this model, then analyze its embeddings to see if we can recover the original biophysical parameters and governing equations — a "discoverative AI" approach. The architecture choices should keep this in mind (e.g., embeddings should be extractable, attention maps should be inspectable).

## What to build

1. `data.py` — Load the HDF5 file. Normalize voltage traces and parameter vectors to zero mean, unit variance. Return a PyTorch Dataset that yields `(parameter_vector, voltage_trace)` pairs.

2. `model.py` — A transformer that takes a 26-dimensional parameter vector as conditioning and autoregressively generates a voltage waveform. Specify and justify:
   - How to tokenize the continuous voltage trace (patch embeddings, binning, or raw values)
   - How to inject the conditioning vector
   - Layer count, hidden dimension, heads, context length
   - The model should be small enough to train on 1000 examples without massive overfitting

3. A short `test_model.py` that instantiates the dataset and model, runs one forward pass with a random batch, and prints the output shape. This confirms everything wires up correctly before training.

## Code style

- One concept per file. No file longer than ~200 lines.
- No unnecessary abstractions. Plain functions and simple classes. No base classes, registries, or config systems.
- Explicit over clever. Write out arguments by name. No `*args/**kwargs` forwarding.
- Inline comments explaining *why*, not *what*.
- Flat project structure. All files in one directory.
- Minimal dependencies: PyTorch, NumPy, h5py. Nothing else.
- Hardcoded defaults at the top of each file. No YAML configs.
- Plain names. `WaveformTransformer`, not `ConditionalAutoRegressiveTemporalModel`.
