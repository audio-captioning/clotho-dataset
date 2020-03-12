#!/usr/bin/env python
# -*- coding: utf-8 -*-

from multiprocessing import Pool

from numpy import rec, dtype

from tools.file_io import dump_numpy_object, load_numpy_object

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['do_sub_function',
           'extract_features_sub']


def do_sub_function(sub_f,
                    it,
                    nb_workers: int):
    if nb_workers > 1:
        with Pool(processes=nb_workers) as pool:
            pool.map(sub_f, it)
    else:
        [sub_f(i) for i in it]


def extract_features_sub(data_file_name,
                         f_func,
                         dir_output_dev,
                         dir_output_eva,
                         settings_features,
                         settings_data):
    """

    :param data_file_name:
    :type data_file_name:
    :param f_func:
    :type f_func:
    :param dir_output_dev:
    :type dir_output_dev:
    :param dir_output_eva:
    :type dir_output_eva:
    :param settings_features:
    :type settings_features:
    :param settings_data:
    :type settings_data:
    :return:
    :rtype:
    """
    data_file = load_numpy_object(data_file_name)

    # Extract the features.
    features = f_func(data_file['audio_data'].item(),
                      **settings_features['process'])

    # Populate the recarray data and dtypes.
    array_data = (data_file['file_name'].item(), )
    dtypes = [('file_name', data_file['file_name'].dtype)]

    # Check if we keeping the raw audio data.
    if settings_features['keep_raw_audio_data']:
        # And add them to the recarray data and dtypes.
        array_data += (data_file['audio_data'].item(), )
        dtypes.append(('audio_data', data_file['audio_data'].dtype))

    # Add the rest to the recarray.
    array_data += (
        features,
        data_file['caption'].item(),
        data_file['caption_ind'].item(),
        data_file['words_ind'].item(),
        data_file['chars_ind'].item())

    dtypes.extend([
        ('features', dtype(object)),
        ('caption', data_file['caption'].dtype),
        ('caption_ind', data_file['caption_ind'].dtype),
        ('words_ind', data_file['words_ind'].dtype),
        ('chars_ind', data_file['chars_ind'].dtype)])

    # Make the recarray
    np_rec_array = rec.array([array_data], dtype=dtypes)

    # Make the path for serializing the recarray.
    parent_path = dir_output_dev \
        if data_file_name.parent.name == settings_data['output_files']['dir_data_development'] \
        else dir_output_eva

    file_path = parent_path.joinpath(data_file_name.name)

    # Dump it.
    dump_numpy_object(np_rec_array, str(file_path))

# EOF
