
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from matplotlib import style
import streamlit as st

style.use('seaborn-v0_8-darkgrid')


def UWB_generate(bandwidth=3.5e9, center_freq=4e9,  duration=1e-9, pulse_amplitude=1, pulse_timing=0, sample_rate=1e14, noise_amplitude=0.02, sample_points=7):
    """
    Generate a UWB Gaussian pulse
    :param bandwidth: Bandwidth of the pulse
    :param center_freq: Center frequency of the pulse
    :param duration: Duration of the pulse
    :param pulse_width: Width of the pulse
    :param pulse_amplitude: Amplitude of the pulse
    :param sample_rate: Sample rate of the pulse
    :return: Pulse
    """
    
    t = np.linspace(-duration/2, duration/2, int(duration * sample_rate))
    uwb_signal = pulse_amplitude * signal.gausspulse(t, fc=center_freq, bw=(bandwidth / center_freq))
    
    original_t = t
    t = t + pulse_timing
    
    spectrum = np.fft.fft(uwb_signal)
    frequencies = np.fft.fftfreq(len(uwb_signal), d=1/sample_rate)
    
    amplitude_spectrum = np.abs(spectrum)
    phase_spectrum = np.angle(spectrum)
    
    real_signal = uwb_signal + np.random.normal(0, noise_amplitude, len(uwb_signal)) * pulse_amplitude
    
    # sample from real signal, get random 7 samples
    sample_indices = np.random.randint(0, len(real_signal), sample_points)
    sample_values = real_signal[sample_indices]
    sample_times = t[sample_indices]
    
    return uwb_signal, t, original_t, spectrum, frequencies, amplitude_spectrum, phase_spectrum, real_signal, sample_values, sample_times


def UWB_FRI_reconstruct(sample_values, sample_times, learning_rate=3):
    """
    Reconstruct the UWB signal from the samples
    :param sample_values: Sampled values
    :param sample_times: Sampled times
    :return: Reconstructed signal
    """
    
    while True:
        center_freq = np.random.uniform(1e9, 8e9)
        bandwidth = np.random.uniform(1e9, 8e9)
        amplitude = np.random.uniform(0.1, 10)
        pulse_timing = np.random.uniform(-1e-9, 1e-9)
        if center_freq - bandwidth / 2 > 0:
            break
    
    t = sample_times
    t = t - pulse_timing
    reconstructed_signal = amplitude * signal.gausspulse(t, fc=center_freq, bw=(bandwidth / center_freq))
    
    # using gradient descent to minimize the error
    
    progress_bar = st.progress(0, text='Reconstructing Signal..., Fitting...')
    
    for i in range(4000):
        error = np.sum(np.abs(reconstructed_signal - sample_values))
        if error < 1e-12:
            break
        
        progress_bar.progress((i / 4000), text='Reconstructing Signal..., Fitting...')
        
        # calculate the center_freq_gradient, bandwidth_gradient, amplitude_gradient, pulse_timing_gradient
        center_freq_gradient = np.sum((reconstructed_signal - sample_values) * signal.gausspulse(t, fc=center_freq, bw=(bandwidth / center_freq)) * 2 * (t) * np.pi * bandwidth / center_freq ** 2)
        bandwidth_gradient = np.sum((reconstructed_signal - sample_values) * signal.gausspulse(t, fc=center_freq, bw=(bandwidth / center_freq)) * 2 * (t) * np.pi * bandwidth / center_freq ** 2)
        amplitude_gradient = np.sum((reconstructed_signal - sample_values) * signal.gausspulse(t, fc=center_freq, bw=(bandwidth / center_freq)))
        pulse_timing_gradient = np.sum((reconstructed_signal - sample_values) * signal.gausspulse(t, fc=center_freq, bw=(bandwidth / center_freq)) * 2 * (t) * np.pi * bandwidth / center_freq ** 2)
        
        
        center_freq -= learning_rate * center_freq_gradient
        bandwidth -= learning_rate * bandwidth_gradient
        amplitude -= learning_rate * amplitude_gradient
        pulse_timing -= learning_rate * pulse_timing_gradient
        
        t = sample_times - pulse_timing
        
        reconstructed_signal = signal.gausspulse(t, fc=center_freq, bw=(bandwidth / center_freq))
        
    progress_bar.progress(100, text='DONE!')
    progress_bar.empty()
    
    return reconstructed_signal, center_freq, bandwidth, amplitude, pulse_timing



if __name__ == '__main__':
    
    sample_rate = 1e14
    
    with st.sidebar:
        
        bandwidth = st.slider('Bandwidth', 1e9, 10e9, 3.5e9, step=0.5e9, format='%e')
        center_freq = st.slider('Center Frequency', 1e9, 10e9, 4e9, step=0.5e9, format='%e')
        duration = st.slider('Duration', 1e-9, 10e-9, 1e-9, step=1e-9, format='%e')
        pulse_amplitude = st.slider('Pulse Amplitude', 0.1, 10.0, 1.0)
        noise_amplitude = st.slider('Noise Amplitude', 0.0, 0.2, 0.02)
        pulse_timing = st.slider('Pulse Timing', -0.5e-9, 0.5e-9, 0.0, format='%e')
        sample_points = st.slider('Number of Sample Points', 3, 20, 7)
        
        
        st.button('Generate UWB Signal and Sample Point Again')

    uwb_signal, t, original_t, spectrum, frequencies, amplitude_spectrum, phase_spectrum, real_signal, sample_values, sample_times = UWB_generate(bandwidth=bandwidth, center_freq=center_freq, duration=duration, pulse_amplitude=pulse_amplitude, pulse_timing=pulse_timing, sample_rate=sample_rate, noise_amplitude=noise_amplitude, sample_points=sample_points)
    
    st.title('UWB Signal Reconstruction')
    st.markdown('This is a demo of UWB signal reconstruction using Gaussian pulse.')
    
    st.markdown('### Generating Signal')
    
    st.write('Bandwidth:', bandwidth / 1e9, 'GHz')
    st.write('Center Frequency:', center_freq / 1e9, 'GHz')
    st.write('Duration:', duration / 1e-9, 'ns')
    st.write('Pulse Amplitude:', pulse_amplitude)
    st.write('Pulse Timing:', pulse_timing / 1e-9, 'ns')
    st.write('Noise Amplitude:', noise_amplitude)
    st.write('Number of Sample Points:', sample_points)
    
    fig = plt.figure()
    plt.plot(t, uwb_signal, color='blue', alpha=1, label='UWB Signal')
    plt.plot(t, real_signal, color='gray', alpha=0.5, label='Real Signal')
    plt.plot(sample_times, sample_values, 'rx', label='Sampled Values')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(visible=True)
    plt.legend()
    
    st.pyplot(fig)
    
    st.markdown('### Reconstructing Signal')
    st.write('Reconstructing the signal from the sampled points. Fitting a Gaussian pulse to the sampled points. The predict parameters:')
    
    reconstructed_signal_sample, center_freq, bandwidth, amplitude, pulse_timing = UWB_FRI_reconstruct(sample_values, sample_times)
    reconstructed_signal = signal.gausspulse(original_t, fc=center_freq, bw=(bandwidth / center_freq))
    reconstructed_t = original_t + pulse_timing
    
    st.write('Bandwidth:', bandwidth / 1e9, 'GHz')
    st.write('Center Frequency:', center_freq / 1e9, 'GHz')
    st.write('Duration:', duration / 1e-9, 'ns')
    st.write('Pulse Amplitude:', amplitude)
    st.write('Pulse Timing:', pulse_timing / 1e-9, 'ns')
    

    fig = plt.figure()
    plt.plot(t, uwb_signal, color='blue', alpha=0.8, label='UWB Signal')
    plt.plot(t, real_signal, color='gray', alpha=0.5, label='Real Signal')
    plt.plot(sample_times, sample_values, 'rx', label='Sampled Values')
    plt.plot(sample_times, reconstructed_signal_sample, 'go', label='Reconstructed Signal Sample')
    plt.plot(reconstructed_t, reconstructed_signal, color='purple', alpha=0.8, label='Reconstructed Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(visible=True)
    plt.legend()
    
    st.pyplot(fig)
    
    
            