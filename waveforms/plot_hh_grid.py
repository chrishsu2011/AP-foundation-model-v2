"""Plot 10 voltage traces from hh_dataset.h5 in a 2x5 grid."""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(42)
idx = np.random.choice(100, 10, replace=False)

with h5py.File("waveforms/hh_dataset_1ms.h5", "r") as f:
    time_s = f["time"][:]          # (1000,) in seconds
    voltage = f["voltage"][sorted(idx)]   # HDF5 requires sorted indices
    amp = f["stimulus/amplitude"][sorted(idx)] # in A/m²

time_ms = time_s * 1e3            # → ms
voltage_mV = voltage * 1e3        # → mV

fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharex=True, sharey=True)

for i, ax in enumerate(axes.flat):
    ax.axvspan(10, 11, color="grey", alpha=0.2) # where stimulus is 
    ax.plot(time_ms, voltage_mV[i], linewidth=0.8)
    ax.set_title(f"#{i}  I={amp[i]:.3f} A/m²", fontsize=9)
    if i >= 5:
        ax.set_xlabel("Time (ms)")
    if i % 5 == 0:
        ax.set_ylabel("V (mV)")

fig.suptitle("Hodgkin–Huxley Voltage Traces", fontsize=13)
fig.tight_layout()
plt.savefig("hh_grid.png", dpi=150)
plt.show()
print("Saved hh_grid.png")
