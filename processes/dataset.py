#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import stdout
from datetime import datetime
from pathlib import Path
from functools import partial

from loguru import logger

from tools.argument_parsing import get_argument_parser
from tools.aux_functions import get_annotations_files, \
    get_amount_of_file_in_dir, check_data_for_split, \
    create_split_data, create_lists_and_frequencies
from tools.file_io import load_settings_file

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['create_dataset']


def create_dataset(settings):
    """Creates the dataset.

    Gets the dictionary with the settings and creates
    the files of the dataset.

    :param settings: Settings to be used.
    :type settings: dict
    """
    # Get logger
    inner_logger = logger.bind(indent=2)

    # Get root dir
    dir_root = Path(settings['directories']['root_dir'])

    # Read the annotation files
    inner_logger.info('Reading annotations files')
    csv_dev, csv_eva = get_annotations_files(
        settings_ann=settings['annotations'],
        dir_ann=dir_root.joinpath(settings['directories']['annotations_dir']))
    inner_logger.info('Done')

    # Get all captions
    inner_logger.info('Getting the captions')
    captions_development = [
        csv_field.get(
            settings['annotations']['captions_fields_prefix'].format(c_ind))
        for csv_field in csv_dev
        for c_ind in range(1, 6)]
    inner_logger.info('Done')

    # Create lists of indices and frequencies for words and characters.
    inner_logger.info('Creating and saving words and chars lists '
                      'and frequencies')
    words_list, chars_list = create_lists_and_frequencies(
        captions=captions_development, dir_root=dir_root,
        settings_ann=settings['annotations'],
        settings_cntr=settings['counters'])
    inner_logger.info('Done')

    # Aux partial function for convenience.
    split_func = partial(
        create_split_data,
        dir_root=dir_root,
        words_list=words_list, chars_list=chars_list,
        settings_ann=settings['annotations'],
        settings_audio=settings['audio'],
        settings_output=settings['output_files'])

    # For each data split (i.e. development and evaluation)
    for split_data in [(csv_dev, 'development'), (csv_eva, 'evaluation')]:

        # Get helper variables.
        split_name = split_data[1]
        split_csv = split_data[0]

        dir_split = dir_root.joinpath(
            settings['output_files']['dir_output'],
            settings['output_files']['dir_data_{}'.format(split_name)])

        dir_downloaded_audio = Path(
            settings['directories']['downloaded_audio_dir'],
            settings['directories']['downloaded_audio_{}'.format(split_name)])

        # Create the data for the split.
        inner_logger.info('Creating the {} split data'.format(split_name))
        split_func(split_csv, dir_split, dir_downloaded_audio)
        inner_logger.info('Done')

        # Count and print the amount of initial and resulting files.
        nb_files_audio = get_amount_of_file_in_dir(
            dir_root.joinpath(dir_downloaded_audio))
        nb_files_data = get_amount_of_file_in_dir(dir_split)

        inner_logger.info('Amount of {} audio files: {}'.format(
            split_name, nb_files_audio))
        inner_logger.info('Amount of {} data files: {}'.format(
            split_name, nb_files_data))
        inner_logger.info('Amount of {} data files per audio: {}'.format(
            split_name, nb_files_data / nb_files_audio))

        # Check the created lists of indices for words and characters.
        inner_logger.info('Checking the {} split'.format(split_name))
        check_data_for_split(
            dir_audio=dir_root.joinpath(dir_downloaded_audio),
            dir_data=Path(settings['output_files']['dir_output'],
                          settings['output_files']['dir_data_{}'.format(
                              split_name)]),
            dir_root=dir_root, csv_split=split_csv,
            settings_ann=settings['annotations'],
            settings_audio=settings['audio'],
            settings_cntr=settings['counters'])
        inner_logger.info('Done')


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
    settings = load_settings_file(args.config_file_dataset)
    main_logger.info('Settings loaded')

    # Create the dataset.
    main_logger.info('Starting Clotho dataset creation')
    create_dataset(settings)
    main_logger.info('Dataset created')


if __name__ == '__main__':
    main()

# EOF
