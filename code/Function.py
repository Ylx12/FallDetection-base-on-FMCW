import random
import numpy as np
import torch


def amplitude_filtering(array, amplitude_threshold):
    for i in range(array.size(0)):
        for j in range(array.size(1)):
            if array[i, j] < amplitude_threshold:
                array[i, j] = 0
    return array


def calculate_barycenter(data, count_threshold, debug):
    temp_barycenter = 0
    count = 0
    for i in range(data.size(0)):
        for j in range(data.size(1)):
            if data[i, j] > 0:
                temp_barycenter = temp_barycenter + i
                count = count + 1
    if count > count_threshold:
        barycenter = temp_barycenter / count
    else:
        barycenter = 0

    if debug:
        print('count=', count, 'barycenter=', barycenter)
    return barycenter, count

def FFT_2D(data1, para):
    sigRangeWin1 = np.zeros((para['chirps'], para['samples']), dtype=complex)
    window = np.hanning(para['samples'] + 2)[1:para['samples'] + 1]

    for l in range(0, para['chirps']):
        sigRangeWin1[l] = np.multiply(data1[l, :], window)

    # range fft processing
    sigRangeFFT = np.zeros((para['chirps'], para['samples']), dtype=complex)

    for l in range(0, para['chirps']):
        sigRangeFFT[l] = np.fft.fft(sigRangeWin1[l], para['fft_Range'])

    # Mean cancellation
    avg = np.sum(sigRangeFFT, axis=0) / 128
    for l in range(0, para['chirps']):
        sigRangeFFT[l, :] = sigRangeFFT[l, :] - avg

    # doppler fft processing
    sigDopplerFFT = np.zeros((para['chirps'], para['samples']), dtype=complex)
    for n in range(0, para['samples']):
        sigDopplerFFT[:, n] = np.fft.fftshift(np.fft.fft(sigRangeFFT[:, n], para['fft_Vel']))

    return sigDopplerFFT


def data_to_RDM(path, para, Rx_num):
    in_data1 = np.loadtxt(path, dtype=complex, comments='#', delimiter=None, converters=None, skiprows=0,
                          unpack=False, ndmin=0)
    in_data1 = in_data1.reshape((para['Rx'], para['chirps'], para['samples']))  # (Rxs, chirps, samples)
    sigDopplerFFT = np.zeros(shape=(para['chirps'], para['samples']), )
    for Rx in range(Rx_num):
        sigDopplerFFT = sigDopplerFFT + np.abs(FFT_2D(in_data1[Rx], para))
    sigDopplerFFT_sf = (sigDopplerFFT) / Rx_num

    RDM = sigDopplerFFT_sf.T
    return RDM


def find_last_index(input_list, target):
    indices = []
    for i in range(len(input_list)):
        if input_list[i] == target:
            indices.append(i)
    if indices:
        return indices[-1]
    else:
        return None


def get_para():
    # constant parameters
    light_speed = 3e8
    Start_Frequency = 60.25e9

    Fs = 6 * 10 ** 6  # data2 Sampling frequency
    Sample_rate = Fs

    para = {
        'light_speed': 3e8,
        'numADCBits': 16,  # The number of bits of ADC sampled data
        'start_frequency': 60.25e9,
        'lamda': light_speed / Start_Frequency,
        'Rx': 4,
        'Tx': 1,

        'sample_rate': Sample_rate,
        'sweepslope': 60e12,
        'samples': 128,  # number of samples of one chirp
        'chirps': 128,  # number of chirps of one frame
        'Tchirp': 160e-6,  # us
        'frames': 9000,
        'Tframe': 100,  # ms

        'fft_Range': 128,  # params.samples
        'fft_Vel': 128,  # params.chirps
        'num_crop': 3,
        'max_value': 1e+04  # data WITH IWR6843
    }
    return para


def normalize_to_0_1(arr):
    min_val = torch.min(arr)
    max_val = torch.max(arr)
    normalized_arr = 1 * (arr - min_val) / (max_val - min_val)
    return normalized_arr


def RDM_prepare(RDarray_list, device):
    data_buffer = np.array(RDarray_list, dtype=float)
    data_buffer = torch.from_numpy(data_buffer)
    data_buffer = data_buffer.reshape((16, 1, 62, 50))
    data_buffer = data_buffer.to(torch.float32)
    data_buffer = data_buffer.to(device=device)

    return data_buffer


def seed_setting(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = predict
        loss = - ((1 - self.alpha) * ((1 - pt + 1e-5) ** self.gamma) * (target * torch.log(pt + 1e-5)) + self.alpha * (
                (pt + +1e-5) ** self.gamma) * ((1 - target) * torch.log(1 - pt + 1e-5)))

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
