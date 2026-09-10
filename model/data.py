"""Load the Hodgkin-Huxley dataset from HDF5 and serve it as a PyTorch Dataset.

Each item is a pair: (parameter_vector, voltage_trace)
    parameter_vector : (26,)  normalised biophysical parameters
    voltage_trace    : (T,)   normalised membrane voltage, T timesteps
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Path is resolved relative to THIS file, not the working directory, so the
# script runs the same whether you launch it from the repo root or from model/.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(THIS_DIR, "..", "waveforms", "hh_dataset_1ms.h5")

# Fixed order for the 26 conditioning values. The model learns a
# position-specific meaning for each entry, so this order must never change
# between training and inference or a checkpoint will silently produce garbage.
PARAMETER_NAMES = [
    "g_Na", "g_K", "g_L",                                    # conductances
    "E_Na", "E_K", "E_L",                                    # reversal potentials
    "Cm",                                                    # capacitance
    "a_m_A", "a_m_Vh", "a_m_k", "b_m_A", "b_m_Vh", "b_m_k",  # Na activation (m)
    "a_h_A", "a_h_Vh", "a_h_k", "b_h_A", "b_h_Vh", "b_h_k",  # Na inactivation (h)
    "a_n_A", "a_n_Vh", "a_n_k", "b_n_A", "b_n_Vh", "b_n_k",  # K activation (n)
    "I_amp",                                                 # stimulus amplitude
]


def load_raw_arrays(path):
    """Read the HDF5 file into plain NumPy arrays (no normalisation yet)."""
    with h5py.File(path, "r") as f:
        voltage = f["voltage"][:].astype(np.float32)   # (N, T) in volts
        time = f["time"][:].astype(np.float32)         # (T,) in seconds

        columns = []
        for name in PARAMETER_NAMES:
            if name == "I_amp":
                # The stimulus amplitude lives in a different HDF5 group than
                # the membrane parameters, but for the model it is just one
                # more number that shapes the waveform.
                columns.append(f["stimulus/amplitude"][:])
            else:
                columns.append(f["parameters"][name][:])
        parameters = np.stack(columns, axis=1).astype(np.float32)  # (N, 26)

    return voltage, parameters, time


class WaveformDataset(Dataset):
    def __init__(self, path=DATASET_PATH):
        voltage, parameters, time = load_raw_arrays(path)

        # Voltage uses ONE global mean/std rather than per-timestep statistics.
        # A spike that occurs at 12 ms and one at 13 ms should look identical
        # to the model; per-timestep normalisation would bake the average
        # spike time into the data and punish any deviation from it.
        self.voltage_mean = float(voltage.mean())
        self.voltage_std = float(voltage.std())

        # Parameters are normalised per column because they live on wildly
        # different scales (g_Na ~ 1200 S/m^2, Cm ~ 0.01 F/m^2). Without this
        # the largest-magnitude parameter would dominate the conditioning.
        self.parameter_mean = parameters.mean(axis=0)
        self.parameter_std = parameters.std(axis=0)
        # A column with zero spread (e.g. if jitter is switched off for one
        # parameter) would divide by zero; leave such columns unscaled.
        self.parameter_std[self.parameter_std == 0.0] = 1.0

        self.voltage = torch.from_numpy(
            (voltage - self.voltage_mean) / self.voltage_std
        )
        self.parameters = torch.from_numpy(
            (parameters - self.parameter_mean) / self.parameter_std
        )
        self.time = time  # kept for plotting; not used by the model

    def __len__(self):
        return self.voltage.shape[0]

    def __getitem__(self, index):
        return self.parameters[index], self.voltage[index]

    def denormalise_voltage(self, normalised_voltage):
        """Undo the scaling so generated traces are back in volts."""
        return normalised_voltage * self.voltage_std + self.voltage_mean
