import numpy as np
from brian2 import *

prefs.codegen.target = "numpy"

EQUATIONS = """
dv/dt = (
    I_amp*stimulus_gate(t)
    - g_Na*m**3*h*(v - E_Na)
    - g_K*n**4*(v - E_K)
    - g_L*(v - E_L)
    ) / Cm : volt

dm/dt = (m_inf - m) / tau_m : 1
dh/dt = (h_inf - h) / tau_h : 1
dn/dt = (n_inf - n) / tau_n : 1

m_inf = am / (am + bm) : 1
tau_m = 1 / (am + bm) : second

h_inf = ah / (ah + bh) : 1
tau_h = 1 / (ah + bh) : second

n_inf = an / (an + bn) : 1
tau_n = 1 / (an + bn) : second

# exprel is used to avoid division by zero near the rate-function threshold.
am = a_m_A*a_m_k/exprel(-(v/mV - a_m_Vh)/a_m_k)/ms : Hz
bm = b_m_A*exp(-(v/mV - b_m_Vh)/b_m_k)/ms : Hz

ah = a_h_A*exp(-(v/mV - a_h_Vh)/a_h_k)/ms : Hz
bh = b_h_A/(1 + exp(-(v/mV - b_h_Vh)/b_h_k))/ms : Hz

an = a_n_A*a_n_k/exprel(-(v/mV - a_n_Vh)/a_n_k)/ms : Hz
bn = b_n_A*exp(-(v/mV - b_n_Vh)/b_n_k)/ms : Hz

# Membrane parameters
g_Na : siemens/meter**2
g_K  : siemens/meter**2
g_L  : siemens/meter**2

E_Na : volt
E_K  : volt
E_L  : volt

Cm    : farad/meter**2
I_amp : amp/meter**2

# Sodium activation
a_m_A  : 1
a_m_Vh : 1
a_m_k  : 1

b_m_A  : 1
b_m_Vh : 1
b_m_k  : 1

# Sodium inactivation
a_h_A  : 1
a_h_Vh : 1
a_h_k  : 1

b_h_A  : 1
b_h_Vh : 1
b_h_k  : 1

# Potassium activation
a_n_A  : 1
a_n_Vh : 1
a_n_k  : 1

b_n_A  : 1
b_n_Vh : 1
b_n_k  : 1
"""

def simulate_batch(batch_size, rng, duration=50 * ms, simulation_dt=0.05 * ms):
    """
    Simulate one batch of randomized Hodgkin-Huxley neurons.

    Returns plain NumPy arrays without Brian2 units.
    """

    # Clear objects from the previous Brian2 simulation.
    start_scope()
    defaultclock.dt = simulation_dt

    num_steps = int(duration / simulation_dt)

    # Shared stimulus timing: on from 10 ms through 40 ms.
    stimulus_values = np.zeros(
        num_steps,
        dtype=np.float32,
    )

    stimulus_values[
        int(10 * ms / simulation_dt):int(11 * ms / simulation_dt)
    ] = 1.0

    stimulus_gate = TimedArray(stimulus_values, dt=simulation_dt,)

    group = NeuronGroup(
        batch_size,
        EQUATIONS,
        method="exponential_euler",
        dt=simulation_dt,
    )

    def multiply_jitter(value, fraction=0.05):
        """Randomly vary a positive parameter around a reference value."""
        return value * rng.uniform(1 - fraction, 1 + fraction, batch_size)

    def additive_jitter(value, spread):
        """Randomly vary a parameter by adding a uniform offset."""
        return value + rng.uniform(-spread, spread, batch_size)

    # Membrane parameters
    group.g_Na = (multiply_jitter(120, 0.10) * msiemens / cm**2)
    group.g_K = (multiply_jitter(36, 0.10) * msiemens / cm**2)
    group.g_L = (multiply_jitter(0.3, 0.10)* msiemens / cm**2)

    group.E_Na = additive_jitter(50, 2) * mV
    group.E_K = additive_jitter(-77, 2) * mV
    group.E_L = additive_jitter(-54.4, 1) * mV

    group.Cm = (multiply_jitter(1.0, 0.05)* ufarad / cm**2)
    group.I_amp = (rng.uniform(8, 14, batch_size)* uamp / cm**2)

    # Rate-function parameters
    group.a_m_A = multiply_jitter(0.1)
    group.a_m_Vh = additive_jitter(-40, 1)
    group.a_m_k = multiply_jitter(10)

    group.b_m_A = multiply_jitter(4.0)
    group.b_m_Vh = additive_jitter(-65, 1)
    group.b_m_k = multiply_jitter(18)

    group.a_h_A = multiply_jitter(0.07)
    group.a_h_Vh = additive_jitter(-65, 1)
    group.a_h_k = multiply_jitter(20)

    group.b_h_A = multiply_jitter(1.0)
    group.b_h_Vh = additive_jitter(-35, 1)
    group.b_h_k = multiply_jitter(10)

    group.a_n_A = multiply_jitter(0.01)
    group.a_n_Vh = additive_jitter(-55, 1)
    group.a_n_k = multiply_jitter(10)

    group.b_n_A = multiply_jitter(0.125)
    group.b_n_Vh = additive_jitter(-65, 1)
    group.b_n_k = multiply_jitter(80)

    # Initial conditions
    group.v = -65 * mV
    group.m = "m_inf"
    group.h = "h_inf"
    group.n = "n_inf"

    monitor = StateMonitor(
        group,
        "v",
        record=True,
        dt=simulation_dt,
    )

    # Explicit Network avoids carrying objects between batches.
    network = Network(group, monitor)

    network.run(
        duration,
        namespace={
            "stimulus_gate": stimulus_gate,
        },
    )

    parameters = {
        "g_Na": np.asarray(
            group.g_Na / (siemens / meter**2),
            dtype=np.float32,
        ),
        "g_K": np.asarray(
            group.g_K / (siemens / meter**2),
            dtype=np.float32,
        ),
        "g_L": np.asarray(
            group.g_L / (siemens / meter**2),
            dtype=np.float32,
        ),
        "E_Na": np.asarray(
            group.E_Na / volt,
            dtype=np.float32,
        ),
        "E_K": np.asarray(
            group.E_K / volt,
            dtype=np.float32,
        ),
        "E_L": np.asarray(
            group.E_L / volt,
            dtype=np.float32,
        ),
        "Cm": np.asarray(
            group.Cm / (farad / meter**2),
            dtype=np.float32,
        ),
        "a_m_A": np.asarray(group.a_m_A, dtype=np.float32),
        "a_m_Vh": np.asarray(group.a_m_Vh, dtype=np.float32),
        "a_m_k": np.asarray(group.a_m_k, dtype=np.float32),
        "b_m_A": np.asarray(group.b_m_A, dtype=np.float32),
        "b_m_Vh": np.asarray(group.b_m_Vh, dtype=np.float32),
        "b_m_k": np.asarray(group.b_m_k, dtype=np.float32),
        "a_h_A": np.asarray(group.a_h_A, dtype=np.float32),
        "a_h_Vh": np.asarray(group.a_h_Vh, dtype=np.float32),
        "a_h_k": np.asarray(group.a_h_k, dtype=np.float32),
        "b_h_A": np.asarray(group.b_h_A, dtype=np.float32),
        "b_h_Vh": np.asarray(group.b_h_Vh, dtype=np.float32),
        "b_h_k": np.asarray(group.b_h_k, dtype=np.float32),
        "a_n_A": np.asarray(group.a_n_A, dtype=np.float32),
        "a_n_Vh": np.asarray(group.a_n_Vh, dtype=np.float32),
        "a_n_k": np.asarray(group.a_n_k, dtype=np.float32),
        "b_n_A": np.asarray(group.b_n_A, dtype=np.float32),
        "b_n_Vh": np.asarray(group.b_n_Vh, dtype=np.float32),
        "b_n_k": np.asarray(group.b_n_k, dtype=np.float32),
    }

    return {
        "time": np.asarray(
            monitor.t / second,
            dtype=np.float32,
        ),
        "voltage": np.asarray(
            monitor.v / volt,
            dtype=np.float32,
        ),
        "stimulus_waveform": stimulus_values,
        "stimulus_amplitude": np.asarray(
            group.I_amp / (amp / meter**2),
            dtype=np.float32,
        ),
        "parameters": parameters,
    }