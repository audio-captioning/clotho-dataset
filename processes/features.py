#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableMapping, Any
from sys import stdout
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from importlib import import_module
from itertools import chain
from datetime import datetime

import numpy as np
from loguru import logger

from tools.file_io import load_numpy_object, dump_numpy_object, load_settings_file
from tools.argument_parsing import get_argument_parser

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['extract_features']


def _extract_features_sub(data_file_name,
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
        ('features', np.dtype(object)),
        ('caption', data_file['caption'].dtype),
        ('caption_ind', data_file['caption_ind'].dtype),
        ('words_ind', data_file['words_ind'].dtype),
        ('chars_ind', data_file['chars_ind'].dtype)
    ])

    # Make the recarray
    np_rec_array = np.rec.array([array_data], dtype=dtypes)

    # Make the path for serializing the recarray.
    parent_path = dir_output_dev \
        if data_file_name.parent.name == settings_data['output_files']['dir_data_development'] \
        else dir_output_eva

    file_path = parent_path.joinpath(data_file_name.name)

    # Dump it.
    dump_numpy_object(np_rec_array, str(file_path))


def extract_features(settings_data: MutableMapping[str, Any],
                     settings_features: MutableMapping[str, Any]) -> None:
    """Extracts features from the audio data of Clotho.

    :param settings_data: Settings for creating data files.
    :type settings_data: dict[str, T]
    :param settings_features: Settings for feature extraction.
    :type settings_features: dict[str, T]
    """
    # Get the root directory.
    dir_root = Path(settings_data['directories']['root_dir'])

    # Get the directories of files.
    dir_output = dir_root.joinpath(settings_data['output_files']['dir_output'])
    dir_dev = dir_output.joinpath(
        settings_data['output_files']['dir_data_development'])
    dir_eva = dir_output.joinpath(
        settings_data['output_files']['dir_data_evaluation'])

    # Get the feature extraction module.
    module_f_func = import_module(
        '.{}'.format(settings_features['module']),
        package=settings_features['package'])

    # Get the feature extraction function.
    f_func = getattr(module_f_func, 'feature_extraction')

    # Get the directories for output.
    dir_output_dev = dir_root.joinpath(
            settings_features['output']['dir_output'],
            settings_features['output']['dir_development'])
    dir_output_eva = dir_root.joinpath(
            settings_features['output']['dir_output'],
            settings_features['output']['dir_evaluation'])

    # Create the directories.
    dir_output_dev.mkdir(parents=True, exist_ok=True)
    dir_output_eva.mkdir(parents=True, exist_ok=True)

    # Apply the function to each file and save the result.
    for data_file_name in filter(lambda _x: _x.suffix == settings_features['data_files_suffix'],
                                 chain(dir_dev.iterdir(), dir_eva.iterdir())):

        # Load the data file.
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
            ('features', np.dtype(object)),
            ('caption', data_file['caption'].dtype),
            ('caption_ind', data_file['caption_ind'].dtype),
            ('words_ind', data_file['words_ind'].dtype),
            ('chars_ind', data_file['chars_ind'].dtype)
        ])

        # Make the recarray
        np_rec_array = np.rec.array([array_data], dtype=dtypes)

        # Make the path for serializing the recarray.
        parent_path = dir_output_dev \
            if data_file_name.parent.name == settings_data['output_files']['dir_data_development'] \
            else dir_output_eva

        file_path = parent_path.joinpath(data_file_name.name)

        # Dump it.
        dump_numpy_object(np_rec_array, str(file_path))


def main():

    # Treat the logging.
    logger.remove()
    logger.add(stdout, format='{level} | [{time:HH:mm:ss}] {name} -- {message}.',
               level='INFO', filter=lambda record: record['extra']['indent'] == 1)
    logger.add(stdout, format='  {level} | [{time:HH:mm:ss}] {name} -- {message}.',
               level='INFO', filter=lambda record: record['extra']['indent'] == 2)
    main_logger = logger.bind(indent=1)

    args = get_argument_parser().parse_args()

    main_logger.info('Doing only dataset creation')

    # Check for verbosity.
    if not args.verbose:
        main_logger.info('Verbose if off. Not logging messages')
        logger.disable('__main__')
        logger.disable('processes')

    main_logger.info(datetime.now().strftime('%Y-%m-%d %H:%M'))

    # Load settings file.
    main_logger.info('Loading settings')
    settings_dataset = load_settings_file(args.config_file_dataset)
    settings_features = load_settings_file(args.config_file_features)
    main_logger.info('Settings loaded')

    # Create the dataset.
    main_logger.info('Starting feature extraction')
    extract_features(settings_data=settings_dataset,
                     settings_features=settings_features)
    main_logger.info('Features extracted')


if __name__ == '__main__':
    main()

# EOF
