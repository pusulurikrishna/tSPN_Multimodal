import numpy as np
import matplotlib.pyplot as plt

def model_epsc_ipsc(rate, duration, amplitude, tau_rise, tau_decay, inward=True, step_size=1):
    """
    Models EPSCs and IPSCs using a Poisson distribution. This function simulates the
    occurrence of excitatory postsynaptic currents (EPSCs) or inhibitory
    postsynaptic currents (IPSCs) based on a given average firing rate
    and applies a rise and decay time course to model the current waveform.

    Args:
        rate (float): The average firing rate of the presynaptic neuron (in Hz).
        duration (float): The duration of the simulation (in ms).
        amplitude (float): The amplitude of each individual EPSC/IPSC (in pA).
        tau_rise (float): Time constant for the rising phase of the EPSC/IPSC (in ms).
        tau_decay (float): Time constant for the decaying phase of the EPSC/IPSC (in ms).
        inward (bool, optional): Boolean indicating whether the current is inward (EPSC, True)
                                or outward (IPSC, False). Defaults to True.
        step_size (float, optional): The time step for the simulation (in ms). Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - time (numpy.ndarray): A numpy array representing the time points (in ms).
            - current (numpy.ndarray): A numpy array representing the EPSC/IPSC waveform.

    Example:
        >>> time, epsc = model_epsc_ipsc(rate=10, duration=5000, amplitude=20,
        ...                             tau_rise=2, tau_decay=8, inward=True, step_size=0.5)
        >>> plt.plot(time, epsc)
        >>> plt.show()
    """

    # Generate time points with the specified step_size
    time = np.arange(0, duration, step_size)

    # Calculate the probability of an EPSC/IPSC occurring in a single time step
    prob_event = rate * step_size / 1000

    # Generate a sequence of random numbers between 0 and 1
    random_numbers = np.random.rand(len(time))

    # Determine the indices where events occur based on the probability
    event_indices = np.where(random_numbers < prob_event)[0]

    # Create an array to store the EPSC/IPSC waveform
    current = np.zeros(len(time))

    # Add the amplitude at the event indices
    current[event_indices] += amplitude

    # Create the EPSC/IPSC kernel with rise and decay phases
    t_kernel = np.arange(0, 5*(tau_rise+tau_decay), step_size)  # Time vector for the kernel with step_size
    kernel = np.exp(-t_kernel / tau_decay) - np.exp(-t_kernel / tau_rise)

    # Convolve with the kernel
    current = np.convolve(current, kernel, mode='same')

    # Adjust the sign of the current based on the 'inward' parameter
    if inward:
        current *= -1  # Inward currents (EPSCs) are negative

    return time, current

if __name__ == "__main__":
    # Example usage - EPSC (inward current)
    rate = 3  # Hz
    duration = 5000  # ms
    amplitude = 20  # pA
    tau_rise = 2  # ms
    tau_decay = 8  # ms
    step_size = 0.5  # ms

    # time, epsc = model_epsc_ipsc(rate, duration, amplitude, tau_rise, tau_decay, inward=True, step_size=step_size)

    # Example usage - IPSC (outward current)
    # time, ipsc = model_epsc_ipsc(rate, duration, amplitude, tau_rise, tau_decay, inward=False, step_size=step_size)

    time, epsc = model_epsc_ipsc(rate=3, duration=20000, amplitude=80, tau_rise=2, tau_decay=100, inward=False,
                                     step_size=0.01)
    # print(ipsc.shape, time.shape)

    # Plot the EPSCs and IPSCs
    plt.plot(time, epsc, label='EPSC')
    # plt.plot(time, ipsc, label='IPSC')
    plt.xlabel('Time (ms)')
    plt.ylabel('Current Amplitude (pA)')
    plt.title('Simulated EPSCs and IPSCs')
    plt.legend()
    plt.show()