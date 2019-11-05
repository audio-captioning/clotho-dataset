#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from librosa.feature import melspectrogram

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['feature_extraction']


def feature_extraction(audio_data, sr, nb_fft, hop_size, nb_mels, f_min, f_max, htk,
                       power, norm, window_function, center):
    """Feature extraction function.

    :param audio_data: Audio signal.
    :type audio_data: numpy.ndarray
    :param sr: Sampling frequency.
    :type sr: int
    :param nb_mels: Amount of MEL bands.
    :type nb_mels: int
    :param nb_fft: Amount of FFT points.
    :type nb_fft: int
    :param hop_size: Hop size in samples.
    :type hop_size: int
    :param f_min: Minimum frequency in Hertz for MEL band calculation.
    :type f_min: float
    :param f_max: Maximum frequency in Hertz for MEL band calculation.
    :type f_max: float|None
    :param htk: Use the HTK Toolbox formula instead of Auditory toolkit.
    :type htk: bool
    :param norm: Area normalization of MEL filters.
    :type norm: bool
    :param window_function: Window function.
    :type window_function: str
    :param center: Center the frame for FFT.
    :type center: bool
    :param power: Power of the magnitude.
    :type power: float
    :return: Log mel-bands energies of shape=(t, nb_mels)
    :rtype: numpy.ndarray
    """
    y = audio_data/abs(audio_data).max()
    mel_bands = melspectrogram(
        y=y, sr=sr, n_fft=nb_fft, hop_length=hop_size, win_length=nb_fft,
        window=window_function, center=center, power=power, n_mels=nb_mels,
        fmin=f_min, fmax=f_max, htk=htk, norm=norm).T

    return np.log(mel_bands + np.finfo(float).eps)

# EOF
