import torch
import copy
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

from obspy.signal.filter import bandpass
from datetime import datetime

tau_len = 15  # correlation shifting length - investigation depth
sampling_rate = 50  # resampling rate of data
dt = 1 / sampling_rate
shift = round(tau_len/dt)
tau = np.arange(shift+1) * dt

def normalize_by_trace(x):
    x_max = x.abs().amax(dim=-1)
    return x / x_max.view(-1, 1)

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def smooth(a, WSZ, usecupy=True):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    if type(a) == torch.Tensor:
        a = a.numpy()
    
    if usecupy:
        out0 = cp.convolve(a, cp.ones(WSZ, dtype=int), 'valid') / WSZ    
        r = cp.arange(1, WSZ-1, 2)
        start = cp.cumsum(a[:WSZ-1])[::2] / r
        stop = (cp.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
        out = cp.concatenate((start, out0, stop))
    else:
        out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ    
        r = np.arange(1, WSZ-1, 2)
        start = np.cumsum(a[:WSZ-1])[::2] / r
        stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
        out = np.concatenate((start, out0, stop))
        
    return out

def temporal_normalization(raw, dt, agcwindow=5):
    """
    record: numpy array (nt,)
    """
    nt = round(agcwindow/2/dt)
    WSZ = 2 * nt + 1
    g = smooth(np.abs(raw), WSZ, usecupy=False)
    return raw / g

def section_plot(record, xs, ts, title, scale=0.3, 
                 save=False, figname=None, t0=None, t1=None,
                 figsize=(20, 8), fillcolors=(plt.get_cmap("coolwarm")(1.0), plt.get_cmap("coolwarm")(0.0)), alpha=0.8):
    """
    record: (nstation, nt)
    """
    record = copy.deepcopy(record)  # or record will be modified

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(record.size(0)):
        x = xs[i] + record[i] * scale
        if fillcolors[0] != None:
            ax.fill_betweenx(ts, x, xs[i], where=(x>xs[i]), color=fillcolors[0])
        if fillcolors[1] != None:
            ax.fill_betweenx(ts, x, xs[i], where=(x<xs[i]), color=fillcolors[1])
        ax.plot(x, ts, 'k', linewidth=1, alpha=alpha)

    ax.set_xlabel("Along-profile distance (km)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Time (s)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(xs[0]-1, xs[-1]+1)
    if t0 is None:
        t0 = ts[0]
        
    if t1 is None: 
        t1 = ts[-1]

    ax.set_ylim(t0, t1)
    ax.invert_yaxis()
    ax.tick_params(labelsize=12, width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_weight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    if save:
        if figname is not None:
            plt.savefig(figname, dpi=600, bbox_inches='tight', format='png')
        else:
            plt.savefig(title+'.png', dpi=600, bbox_inches='tight', format='png')

def data_processing(data_src, data_rec, freqmin=0.1, freqmax=1, whiten_window=0.5, fs=sampling_rate, shift=shift):
    '''
    data_...: (nt,), numpy array
    '''
    data_src = data_src - data_src.mean()
    data_rec = data_rec - data_rec.mean()
    data_src = bandpass(data_src, freqmin=freqmin, freqmax=freqmax, df=fs, zerophase=True)
    data_rec = bandpass(data_rec, freqmin=freqmin, freqmax=freqmax, df=fs, zerophase=True)
    data_src = temporal_normalization(data_src, dt, agcwindow=5)
    data_rec = temporal_normalization(data_rec, dt, agcwindow=5)
    
    data_src, data_rec = cp.array(data_src), cp.array(data_rec)
    # whitening + correlation in the frequency domain 
    # standardize to make correlation lie between -1 and 1
    data_src = (data_src - data_src.mean()) / (data_src.std() * len(data_src))
    data_rec = (data_rec - data_rec.mean()) / data_rec.std() 
    signal_length = len(data_src)
    x_cor_sig_length = signal_length * 2 + 1
    fast_length = nextpow2(x_cor_sig_length)
    fft_src = cp.fft.rfft(data_src, fast_length, axis=-1)
    fft_rec = cp.fft.rfft(data_rec, fast_length, axis=-1)
    fft_multiplied = cp.conj(fft_src) * fft_rec
    freqs = cp.arange(fast_length // 2 + 1) * fs / (fast_length - 1)
    df = freqs[1] - freqs[0]
    nf = int((whiten_window/2/df).item())
    WSZ = 2 * nf + 1
    g_src = smooth(cp.abs(fft_src), WSZ)
    g_rec = smooth(cp.abs(fft_rec), WSZ)
    whitened = fft_multiplied / (g_src * g_rec)
    if freqmin is not None:
        whitened[freqs<freqmin] = 0
    if freqmax is not None:
        whitened[freqs>freqmax] = 0
    prelim_corr = cp.fft.irfft(whitened, axis=-1)  
    truncate = shift * 2 + 1
    final_corr = cp.roll(prelim_corr, fast_length//2, axis=-1)[fast_length//2-truncate//2:fast_length//2-truncate//2+truncate]
    final_corr = final_corr[shift:] + final_corr[:shift+1][::-1]
    final_corr = final_corr / cp.abs(final_corr).max()
    return cp.asnumpy(final_corr)

def sort_datetime(date_strings):
    # Convert date strings to datetime objects
    date_objects = [datetime.strptime(date, '%Y%m%d%H') for date in date_strings]
    # Sort the datetime objects
    date_objects.sort()
    # Convert sorted datetime objects back to the original string format
    sorted_date_strings  = [date.strftime('%Y%m%d%H') for date in date_objects]
    return sorted_date_strings