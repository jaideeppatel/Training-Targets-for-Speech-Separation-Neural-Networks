import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import seaborn as sns


def hann(x, N):
    """
    Defining generic hann window
    :param x:
    :param N:
    :return:
    """
    return (1 - np.cos(2 * np.pi * x / (N - 1))) / 2


def generate_dft_coeffs(N):
    """
    Funtion to Generate  complex Fourier Coefficients
    :param N: frame size
    :return: Complex fourier coefficients
    """
    F = np.zeros((N, N), dtype=complex)
    # F_cos = np.zeros((N, N))
    # F_sin = np.zeros((N, N))

    for f in range(N):
        for n in range(N):
            F[f, n] = np.cos(2 * np.pi * f * n / N) + (np.sin(2 * np.pi * f * n / N)) * 1j
            # F_cos[f, n] = np.cos(2 * np.pi * f * n / N)
            # F_sin[f, n] = np.sin(2 * np.pi * f * n / N)

    # return F_cos + 1j * F_sin
    return F


def generate_data_samples(data, N):
    """
    NOTE: Overlapping factor is hardcoded to 0.5

    Segmentation of sample stream to blocks of size N
    and overalap between segments is N/2
    :param data: Time series data
    :param N: Frame size
    :return: Segment data into overlapping frames and return a matrix with each column corresponding to a segment
    """

    data_len = data.shape[0]

    X = [data[i:i + N] for i in range(0, data_len - N // 2, N // 2)]
    if len(X[-1]) != N:
        X[-1] = np.append(X[-1], np.zeros(N - X[-1].shape[0]))

    return np.array(X).transpose()


def reconstruct_signal(X):
    """
    Reconstruct signal from segmented matrix
    :param X: segmentized matrix
    :return: reconstructed time series signal
    """

    width = X.shape[1]
    N = X.shape[0]
    n = N // 2

    head = X[:n, 0]
    tail = X[n:, width - 1]

    body = np.array([X[n:, i] + X[:n, i + 1] for i in range(width - 1)]).reshape(n * (width - 1))

    return np.append(head, np.append(body, tail))


def get_spectrogram(data, fft_coeffs):
    """
    Default window is hann. Multiple windows customization can be done later usign a function parameter
    :param data: time series data
    :param fft_coeffs: Fouries transforms coefficient (dtype: complex)
    :return: Complex spectrogram
    """

    N = fft_coeffs.shape[0]

    X_raw = generate_data_samples(data, N)
    hann_win = hann(np.array(range(N), ndmin=2), N)

    X = np.multiply(X_raw, hann_win.T)

    spectrogram = np.dot(fft_coeffs, X)

    return spectrogram


def get_wave(spectrogram, fft_coeffs, is_full_spectrogram=False):
    """
    Takes full or upper half of the spectrogram and returns reconstructed time series signal
    :param spectrogram: spectrogram
    :param fft_coeffs: complex fourier coefficients
    :param is_full_spectrogram: default False, if true, the full spectrogram is used
    :return: time series signal
    """

    # in half spectrogram, 1st row is not in mirror image
    # last line row is same, so, we flip 2nd to last but one rows

    if not is_full_spectrogram:
        # generating full spectrogram from the upper half if 'is_full_spectrogram' is False
        full_spectrogram = np.vstack((spectrogram[0],
                                      spectrogram[1:-1],
                                      spectrogram[-1],
                                      np.flipud(spectrogram[1:-1]).conjugate()
                                      ))
    else:
        full_spectrogram = spectrogram

    # Doing Fourier inverse to get segmented matrix
    signal_segment_matrix = np.dot(fft_coeffs.conjugate().T, full_spectrogram).real

    # Recovering timeseries signal
    signal = reconstruct_signal(signal_segment_matrix)

    # Rescaling the signal
    # signal = (signal - signal.min()) / (signal.max() - signal.min())
    # signal = signal * 3 / signal.var()

    # returning signal
    return signal


if __name__ == '__main__':
    
    # finding sampling rate
    sampling_rate = 16000

    # setting frame size for STFT
    frame_size = 1024

    # Generating DFT coefficients
    F = generate_dft_coeffs(frame_size)

    
    print('Recovering test signal')
    # recovering test signal spectrum using IBM learnt from test mixture
    S_test_pred = M_test * X_test

    # recovering time signal from spectrum
    s_test_pred = get_wave(S_test_pred, F)

    # To make sure recovered signal is of the same length as test signal
#    assert s_test_pred.shape == s_test.shape

    print('Writing recovered signal to file')
    # writing recovered signal to output folder
    # with out any normalization
    wav.write(data=s_test_pred, rate=sampling_rate, filename='s_cap.wav')

    # Normalizing with variance
    s_test_pred_normalized = s_test_pred / s_test_pred.var()
    # writing to output folder
    wav.write(data=s_test_pred_normalized, rate=sampling_rate, filename='s_cap_norm.wav')

    # SNR Calculation
    print('Computing SNR')
    print('\n\n', '-'*20, 'SNR', '-'*20)
    s_test = s_test / s_test.max()
    s_test_pred = s_test_pred / s_test_pred.var()
    s_test_pred_normalized = s_test_pred_normalized / s_test_pred_normalized.max()

    SNR = 10 * np.log10(s_test.var() / (s_test - s_test_pred).var())
    print('SNR of recovered signal:', SNR)

