#!/usr/bin/env python

# Copyright Brno University of Technology
# Licensed under the Apache License, Version 2.0 (the "License")
# @Authors: Lukas Burget, Katerina Zmolikova
# @Emails: burget@fit.vutbr.cz, izmolikova@fit.vutbr.cz

import torch
import numpy as np
from features import mel_inv, mel, povey_window, mel_fbank_mx

def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) // shift + 1, window) + a.shape[1:]
    strides = (a.stride()[0]*shift,a.stride()[0]) + a.stride()[1:]
    return torch.as_strided(a, size = shape, stride = strides)

def preemphasis(x, coef=0.97):
    return x - torch.cat((x[..., :1], x[..., :-1]), dim = -1) * coef

def fbank_htk(x, window, noverlap, fbank_mx, nfft=None, _E=None,
             USEPOWER=False, RAWENERGY=True, PREEMCOEF=0.97, ZMEANSOURCE=False,
             ENORMALISE=True, ESCALE=0.1, SILFLOOR=50.0, USEHAMMING=True):
    """Mel log Mel-filter bank channel outputs
    Returns NUMCHANS-by-M matrix of log Mel-filter bank outputs extracted from
    signal x, where M is the number of extracted frames, which can be computed
    as floor((length(x)-noverlap)/(window-noverlap)). Remaining parameters
    have the following meaning:
    x         - input signal
    window    - frame window length (in samples, i.e. WINDOWSIZE/SOURCERATE) 
                or vector of window weights override default windowing function
                (see option USEHAMMING)
    noverlap  - overlapping between frames (in samples, i.e window-TARGETRATE/SOURCERATE)
    fbank_mx  - array with (Mel) filter bank (as returned by function mel_fbank_mx()).
                Note that this must be compatible with the parameter 'nfft'.
    nfft      - number of samples for FFT computation. By default, it is set in the
                HTK-compatible way to the window length rounded up to the next higher
                power of two.
    _E        - include energy as the "first" or the "last" coefficient of each
                feature vector. The possible values are: "first", "last", None.

    Remaining options have exactly the same meaning as in HTK.

    See also:
      mel_fbank_mx:
          to obtain the matrix for the parameter fbank_mx
      add_deriv: 
          for adding delta, double delta, ... coefficients
      add_dither:
          for adding dithering in HTK-like fashion
    """
    if type(USEPOWER) == bool:
        USEPOWER += 1
    if np.isscalar(window):
        window = np.hamming(window) if USEHAMMING else np.ones(window)
    if nfft is None:
        nfft = 2**int(np.ceil(np.log2(window.size)))
    x = framing(x.double(), window.size, window.size-noverlap)
    if ZMEANSOURCE:
        x = x - x.mean(dim=1)[:,None]
    if _E is not None and RAWENERGY:
        energy = torch.log((x**2).sum(dim=1))
    if PREEMCOEF is not None:
        x = preemphasis(x, PREEMCOEF)
    x *= torch.tensor(window).to(x.device)
    if _E is not None and not RAWENERGY:
        energy = torch.log((x**2).sum(dim=1))
    #x = np.abs(scipy.fftpack.fft(x, nfft))
    #x = x[:,:x.shape[1]/2+1]
    x_padded = torch.cat((x, torch.zeros(x.shape[0],
                                        nfft - window.size).double().to(x.device)),
                        dim = 1)
    x = torch.rfft(x_padded, 1)
    #x = np.abs(x)
    x = x[...,0]**2 + x[...,1]**2
    if USEPOWER != 2:
        x = x ** (0.5 * USEPOWER)
    fbank_mx = torch.tensor(fbank_mx).to(x.device)
    x = torch.log(torch.clamp(torch.matmul(x, fbank_mx), min = 1.0))
    if _E is not None and ENORMALISE:
        energy = (energy - energy.max())       * ESCALE + 1.0
        min_val  = -np.log(10**(SILFLOOR/10.)) * ESCALE + 1.0
        energy[energy < min_val] = min_val

    res = torch.cat(([energy[:,None]] if _E == "first" else []) + [x] +
                     ([energy[:,None]] if (_E in ["last", True])  else []))

    return res

def add_dither(x, level=8):
    return x + level * (torch.rand(*x.shape)*2-1).to(x.device)

def cmvn_floating_kaldi(x, LC,RC, norm_vars=True):
    """Mean and variance normalization over a floating window.
    x is the feature matrix (nframes x dim)
    LC, RC are the number of frames to the left and right defining the floating
    window around the current frame. This function uses Kaldi-like treatment of
    the initial and final frames: Floating windows stay of the same size and
    for the initial and final frames are not centered around the current frame
    but shifted to fit in at the beginning or the end of the feature segment.
    Global normalization is used if nframes is less than LC+RC+1.
    """
    N, dim = x.shape
    win_len = min(len(x),  LC+RC+1)
    win_start = np.maximum(np.minimum(np.arange(-LC,N-LC), N-win_len), 0)
    f = torch.cat((torch.zeros((1, dim)).to(x.device).type(x.dtype), torch.cumsum(x, 0)))
    # print((f[win_start+win_len]-f[win_start])/win_len)
    x = x - (f[win_start+win_len]-f[win_start])/win_len
    if norm_vars:
      f = torch.cat((torch.zeros((1, dim)).to(x), torch.cumsum(x**2, 0)))
      x /= torch.sqrt((f[win_start+win_len]-f[win_start])/win_len)
    return x

def calculate_features(signal, noverlap=240, winlen=400, LC=150, RC=149, fs=16000):
    """Wrapper for the feature extraction"""
    window = povey_window(winlen)
    signal = add_dither(signal * 32768)
    # Mirror noverlap//2 initial and final samples
    signal = torch.cat((torch.flip(signal[:noverlap//2-1], (0,)),
                        signal,
                        torch.flip(signal[-winlen//2-1:], (0,))))
    fbank_mx = mel_fbank_mx(winlen, fs, NUMCHANS=40,
                            LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
    fea = fbank_htk(signal, window, noverlap, fbank_mx,
                    USEPOWER=True, ZMEANSOURCE=True)
    fea = cmvn_floating_kaldi(fea, LC, RC, norm_vars=False)
    return fea
