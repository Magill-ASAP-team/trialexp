from kilosort.io import *
from kilosort.run_kilosort import get_run_parameters
from scipy import signal
import torchaudio
import matplotlib.pylab as plt

def cascade_filter_coeff(blist, alist):
    # combine the filter coefficient together 
    # multiplication in freq. domain is equivalent to convolution in time domain
    b = blist[0]
    a = alist[0]

    for i in range(1, len(blist)):
        b = np.convolve(b, blist[i])
        a = np.convolve(a, alist[i])

    return b,a

def find_artifacts_freq(x, nperseg, freq_cutoff, q_factor, verbose=False, Fs=30000):

    f,pxx = signal.welch(x, fs=Fs, nperseg=nperseg)

    # remove the baseline of the power spectrum
    p = np.polyfit(f, pxx,10)
    fitted_p = np.polyval(p, f)
    pxxs = pxx/fitted_p - 1

    peak_idx = signal.find_peaks(pxxs, height=0.15, distance=10)[0]
    peak_f = f[peak_idx]

    freq_notch = peak_f[peak_f>freq_cutoff]
    # combine the filter for efficiency
    alist = []
    blist = []
    for freq in freq_notch:
        [b,a] = signal.iirnotch(freq, q_factor, Fs)
        blist.append(b)
        alist.append(a)
    b,a = cascade_filter_coeff(blist,alist)

    # for verification
    if verbose:
        fig,ax = plt.subplots(1,2,figsize=(3*3,3))
        ax[0].semilogy(f,pxx)
        ax[0].semilogy(f[peak_idx],pxx[peak_idx],'.')
        ax[0].set_title('Original spectrum')

        w,h = signal.freqz(b,a)
        ax[1].plot(w,h)
        ax[1].set_title('Frequency response of filter')
        
        print(f'I will do notch filtering for the following frequencies: {freq_notch}')

    return freq_notch, b, a

def remove_sensor_artifact_torch(x, channel2analyze=0, nperseg=2046,freq_cutoff=2800, q_factor=30, verbose=False, Fs=30000, device='cpu'):
    # attempt to remove the sensor artifact by identify the peaks in power spectrum and apply notch filter to the signal
    # x should be in channel x time format, it should be a torch Tensor

    freq_notch,b,a = find_artifacts_freq(x[channel2analyze,:].cpu(), nperseg, freq_cutoff, q_factor, verbose)
    b = torch.tensor(b).to(device, dtype=torch.float)
    a = torch.tensor(a).to(device, dtype=torch.float)
    x_tensor = x.to(device)
    # print(x_tensor.dtype, b.dtype, a.dtype)
    factor = x_tensor.abs().max()
    x_tensor /= factor

    x_tensor = torchaudio.functional.filtfilt(x_tensor, a, b, clamp=False)
    x_tensor *= factor # get back to the original amplitude
    
    # signal_filt = cusignal.filtfilt(b,a,x_tensor) 
    #conversion to and from torch tensor from cupy is quite buggy, not using it for now
    
    return x_tensor

def filter_artifact_sensor(self, X, ops=None, ibatch=None):
    # filter artifact that is contamineted by the cap sensor, only use when the artifact is strong
    # pick only the channels specified in the chanMap
    # if ibatch == 300:
    #     plt.figure()
    #     plt.plot(X[0,:].cpu().numpy())
        
    if self.chan_map is not None:
        X = X[self.chan_map]

    if self.invert_sign:
        X = X * -1

    X = X - X.mean(1).unsqueeze(1)


    
    if self.do_CAR:
        # remove the mean of each channel, and the median across channels
        X = X - torch.median(X, 0)[0]

    # high-pass filtering in the Fourier domain (much faster than filtfilt etc)
    if self.hp_filter is not None:
        fwav = fft_highpass(self.hp_filter, NT=X.shape[1])
        X = torch.real(ifft(fft(X) * torch.conj(fwav)))
        X = fftshift(X, dim = -1)

    # Do the artifact removal before any other steps
    # print('removing artifacts')
    print(ibatch)
    X=remove_sensor_artifact_torch(X, freq_cutoff=900, device=self.device)

    # whitening, with optional drift correction
    if self.whiten_mat is not None:
        if self.dshift is not None and ops is not None and ibatch is not None:
            M = get_drift_matrix(ops, self.dshift[ibatch], device=self.device)
            #print(M.dtype, X.dtype, self.whiten_mat.dtype)
            X = (M @ self.whiten_mat) @ X
        else:
            X = self.whiten_mat @ X
    return X