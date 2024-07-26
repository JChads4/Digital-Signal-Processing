import numpy as np
import matplotlib.pyplot as plt
import itertools


######################################## GENERATING A SIGNAL #####################################################

# Pulse model - two component exponential function for a Ge pre-amp signal.
def pulse_model(i, tau1, tau2, height=0, start=0):
    '''
        1. np.zeros((start,)) creates an array of zeros with length equal to the start parameter. 
           This array of zeros is prepended to the pulse signal to introduce a delay.
           The comma after start is used because np.zeros expects a tuple, so this is just a tuple with 1 dimension.

        2. The expression i[:i.size - start] adjusts the indices used in the exponential 
           to account for the delay introduced by start. The pulse calculation 
           np.exp(-i[:i.size - start] / tau1) - np.exp(-i[:i.size - start] / tau2) is only applied to the 
           portion of the array after the leading zeros. This effectively shifts the pulse to start at the desired position.
    '''
    # Create an array of zeros with length 'start'
    leading_zeros = np.zeros((start,))
    
    # Calculate the pulse signal
    pulse_signal = (tau1 / (tau1 - tau2)) * height * (
        np.exp(-i[:i.size - start] / tau1) - np.exp(-i[:i.size - start] / tau2)
    )
    
    # Append the leading zeros to the beginning of the pulse signal
    return np.append(leading_zeros, pulse_signal)


# Create numpy array from function 
def pulse(length, start, height, tau1, tau2):

    '''
        1. np.fromfunction(function, shape, **kwargs) - this creates an array with 'length shape'.

    '''
    # Create an array by applying 'pulse_model' to each index
    return np.fromfunction(pulse_model,(length,), tau1=tau1, tau2=tau2, height=height, start=start).astype(int)

# Signal params
tlen = 100000  # Duration of signal tail
ppos = 50000   # Signal will start at this time value
mvbin = 1.0 / 0.122  # Conversion factor for height
dtau = 5000    # Time constant tau1
rtau = 2       # Time constant tau2

# Generate signal
x = pulse(tlen, ppos, 1.332*200*mvbin, dtau, rtau)

fig, axes = plt.subplots(4, 2) 

# Create empty plots to fill in as the signal processing evolves.
for i in range(8):
    ax = axes[i // 2, i % 2]
    ax.plot([])  # Empty plot

# Add the first graph into the first subplot
axes[0,0].plot(x)
axes[0,0].set_title('Initial Signal')

# Define a noise function to add to the signal
def noise(length, amplitude):
    '''
        1. length: The number of elements in the noise array, which should match the length of the signal array to which it will be added.
        2. Amplitude: The maximum absolute value of the noise. 
            The noise values will be randomly selected integers in the range from -amplitude to amplitude.
    '''
    return np.random.randint(-1*amplitude, amplitude, (length,))

nlev = 2 # This is a scaling factor for the noise amplitude.
x += noise(x.size, nlev*mvbin) # The actual amplitude of the noise is the product of the noise level. This adds the noise to the signal.

axes[1,0].plot(x, label='Signal + noise')
axes[1,0].set_title('Signal + Noise')

# Move to 0V baseline
x += 8192
axes[1,0].plot(x, label='0V Baseline')
axes[1,0].legend()

######################################## HISTOGRAMMING THE PULSE HEIGHT #####################################################

# Lets sum a region near the top of the signal
''' 
    >> 6: This is a bitwise right shift operation. It effectively divides the sum by 2^6 (64).
    Bit shifting is generally faster than multiplication or division because it is directly supported by the hardware at the bit level.
'''
start_index, end_index = 50032, 50096
i = np.sum(x[start_index:end_index]) >> 6 
print(f"Sum between {start_index} - {end_index} = {i}")
axes[2,0].plot(x)
axes[2,0].axvspan(start_index, end_index, color='b', alpha=0.2, label='Summing Region')
axes[2,0].legend()
axes[2,0].set_xlim(start_index-300, end_index+300)
axes[2,0].set_title('Summing for pulse height')

# Create a histogram to get the pulse height
h1 = np.zeros(16384) # histogram with 2^14 bins, that will store counts when they are added.
for i in range(300): # run 300 signals to populate the histogram, each time the summed region will be appended.
    x = pulse(tlen, ppos, 1.332*200*mvbin, dtau, rtau) + noise(x.size, nlev*mvbin) + 8192
    i = np.sum(x[start_index:end_index]) >> 6
    h1[i] +=1

axes[3,0].stairs(h1)
axes[3,0].set_title('Histogram of simulated signal energies using summing')

# Subtract the stuff outside the signal
h2 = np.zeros(16384) # histogram with 2^14 bins, that will store counts when they are added.
for i in range(300): # run 300 signals to populate the histogram, each time the summed region will be appended.
    x = pulse(tlen, ppos, 1.332*200*mvbin, dtau, rtau) + noise(x.size, nlev*mvbin) + 8192
    i = np.sum(x[start_index:end_index]) - np.sum(x[start_index-1000:end_index-1000]) >> 6
    h2[i] +=1

# Plot summed and subtracted regions to verify
axes[0,1].plot(x)
axes[0,1].axvspan(start_index, end_index, color='b', alpha=0.2, label='Summing Region')
axes[0,1].axvspan(start_index-1000, end_index-1000, color='r', alpha=0.2, label='Subtraction Region')
axes[0,1].legend()
axes[0,1].set_xlim(start_index-2000, end_index+300)
axes[0,1].set_title('Signal Subtraction region')

# plot both histograms now
axes[1,1].stairs(h1)
axes[1,1].stairs(h2)
axes[1,1].set_title('Subtracted & Unsubtracted Energy Histogram')

plt.tight_layout()
plt.show()

