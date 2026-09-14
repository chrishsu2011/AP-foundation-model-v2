from pathlib import Path

import h5py
import numpy as np
from brian2 import ms

from hh_waveform_simulation import simulate_batch

# Dataset settings
NUMBER_OF_WAVEFORMS = 10000
BATCH_SIZE = 1000

DURATION = 50 * ms
SIMULATION_DT = 0.01 * ms

RANDOM_SEED = 42

OUTPUT_PATH = Path(__file__).parent / "hh_dataset_1ms.h5"


PARAMETER_UNITS = {
    "g_Na": "siemens/meter**2",
    "g_K": "siemens/meter**2",
    "g_L": "siemens/meter**2",
    "E_Na": "volt",
    "E_K": "volt",
    "E_L": "volt",
    "Cm": "farad/meter**2",
}


def create_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    num_steps = int(DURATION / SIMULATION_DT)

    with h5py.File(OUTPUT_PATH, "w") as file:
        # General metadata
        file.attrs["model"] = "Hodgkin-Huxley"
        file.attrs["number_of_waveforms"] = NUMBER_OF_WAVEFORMS
        file.attrs["batch_size"] = BATCH_SIZE
        file.attrs["random_seed"] = RANDOM_SEED
        file.attrs["duration_ms"] = float(DURATION / ms)
        file.attrs["simulation_dt_ms"] = float(
            SIMULATION_DT / ms
        )

        # Preallocate the large voltage dataset.
        voltage_ds = file.create_dataset(
            "voltage",
            shape=(NUMBER_OF_WAVEFORMS, num_steps),
            dtype=np.float32,
            chunks=(min(BATCH_SIZE, NUMBER_OF_WAVEFORMS), num_steps),
            compression="gzip",
            compression_opts=4,
        )
        voltage_ds.attrs["unit"] = "volt"

        # Time and stimulus groups
        time_ds = file.create_dataset(
            "time",
            shape=(num_steps,),
            dtype=np.float32,
        )
        time_ds.attrs["unit"] = "second"

        stimulus_group = file.create_group("stimulus")

        stimulus_waveform_ds = stimulus_group.create_dataset(
            "waveform",
            shape=(num_steps,),
            dtype=np.float32,
        )
        stimulus_waveform_ds.attrs["unit"] = "dimensionless"

        stimulus_amplitude_ds = stimulus_group.create_dataset(
            "amplitude",
            shape=(NUMBER_OF_WAVEFORMS,),
            dtype=np.float32,
            chunks=(min(BATCH_SIZE, NUMBER_OF_WAVEFORMS),),
            compression="gzip",
            compression_opts=4,
        )
        stimulus_amplitude_ds.attrs["unit"] = "amp/meter**2"

        # Parameter datasets are created after the first batch,
        # once their names are known.
        parameter_group = file.create_group("parameters")
        parameter_datasets = {}

        for start in range(
            0,
            NUMBER_OF_WAVEFORMS,
            BATCH_SIZE,
        ):
            stop = min(
                start + BATCH_SIZE,
                NUMBER_OF_WAVEFORMS,
            )

            current_batch_size = stop - start

            batch = simulate_batch(
                batch_size=current_batch_size,
                rng=rng,
                duration=DURATION,
                simulation_dt=SIMULATION_DT,
            )

            # Initialize values shared by every batch.
            if start == 0:
                time_ds[:] = batch["time"]
                stimulus_waveform_ds[:] = batch[
                    "stimulus_waveform"
                ]

                for name in batch["parameters"]:
                    dataset = parameter_group.create_dataset(
                        name,
                        shape=(NUMBER_OF_WAVEFORMS,),
                        dtype=np.float32,
                        chunks=(
                            min(
                                BATCH_SIZE,
                                NUMBER_OF_WAVEFORMS,
                            ),
                        ),
                        compression="gzip",
                        compression_opts=4,
                    )

                    dataset.attrs["unit"] = (
                        PARAMETER_UNITS.get(
                            name,
                            "dimensionless",
                        )
                    )

                    parameter_datasets[name] = dataset

            # Write this batch directly into its rows.
            voltage_ds[start:stop] = batch["voltage"]

            stimulus_amplitude_ds[start:stop] = batch[
                "stimulus_amplitude"
            ]

            for name, values in batch["parameters"].items():
                parameter_datasets[name][start:stop] = values

            # Ensure the completed batch is written to disk.
            file.flush()

            print(
                f"Saved waveforms {start:,} through "
                f"{stop - 1:,} "
                f"({stop:,}/{NUMBER_OF_WAVEFORMS:,})"
            )

    print(f"\nDataset saved to:\n{OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    create_dataset()

# preview of file structure
"""
hh_dataset.h5
├── time                         shape (T,)
├── voltage                      shape (N, T)
├── stimulus
│   ├── waveform                 shape (T,)
│   └── amplitude                shape (N,)
├── parameters
│   ├── g_Na                     shape (N,)
│   ├── g_K                      shape (N,)
│   ├── g_L                      shape (N,)
│   ├── E_Na                     shape (N,)
│   ├── E_K                      shape (N,)
│   ├── E_L                      shape (N,)
│   ├── Cm                       shape (N,)
│   └── ...rate parameters
├── qc
│   ├── valid                    shape (N,)
│   └── spike_count              shape (N,)
└── split                        shape (N,)
"""