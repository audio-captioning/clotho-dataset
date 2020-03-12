#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableMapping, Any
from sys import stdout
from functools import partial
from pathlib import Path
from importlib import import_module
from itertools import chain
from datetime import datetime

from loguru import logger

from tools.file_io import load_settings_file
from tools.argument_parsing import get_argument_parser
from tools.multi_processing import do_sub_function, extract_features_sub

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['extract_features']


def extract_features(settings_data: MutableMapping[str, Any],
                     settings_features: MutableMapping[str, Any],
                     nb_workers: int) -> None:
    """Extracts features from the audio data of Clotho.

    :param settings_data: Settings for creating data files.
    :type settings_data: dict[str, T]
    :param settings_features: Settings for feature extraction.
    :type settings_features: dict[str, T]
    :param nb_workers: Amount of workers to use. If < 1, then no\
                    multi-process is happening.
    :type nb_workers: int
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

    sub_f = partial(
        extract_features_sub,
        f_func=f_func,
        dir_output_dev=dir_output_dev,
        dir_output_eva=dir_output_eva,
        settings_features=settings_features,
        settings_data=settings_data)

    it = filter(lambda _x: _x.suffix == settings_features['data_files_suffix'],
                chain(dir_dev.iterdir(), dir_eva.iterdir()))

    do_sub_function(sub_f, it, nb_workers)


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
