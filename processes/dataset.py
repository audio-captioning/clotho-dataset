#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import stdout
from datetime import datetime
from pathlib import Path
from functools import partial

from loguru import logger

from tools.argument_parsing import get_argument_parser
from tools.aux_functions import get_annotations_files, \
    check_amount_of_files, check_data_for_split, \
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
    inner_logger.info('Reading annotations files.')
    csv_development, csv_evaluation = get_annotations_files(
        settings_ann=settings['annotations'],
        dir_ann=dir_root.joinpath(settings['directories']['annotations_dir']))
    inner_logger.info('Done.')

    # Get all captions
    inner_logger.info('Getting the captions.')
    captions_development = [
        csv_field.get(settings['annotations']['captions_fields_prefix'].format(c_ind))
        for csv_field in csv_development
        for c_ind in range(1, 6)]
    inner_logger.info('Done.')

    inner_logger.info('Creating and saving words and chars lists and frequencies.')
    words_list, chars_list = create_lists_and_frequencies(
        captions=captions_development, dir_root=dir_root,
        settings_ann=settings['annotations'],
        settings_cntr=settings['counters'])
    inner_logger.info('Done.')

    dir_split_dev = dir_root.joinpath(
        settings['output_files']['dir_output'],
        settings['output_files']['dir_data_development'])

    dir_split_eva = dir_root.joinpath(
        settings['output_files']['dir_output'],
        settings['output_files']['dir_data_evaluation'])

    split_func = partial(
        create_split_data,
        dir_root=dir_root,
        words_list=words_list, chars_list=chars_list,
        settings_ann=settings['annotations'],
        settings_audio=settings['audio'],
        settings_output=settings['output_files']
    )

    dir_downloaded_audio_dev = Path(
        settings['directories']['downloaded_audio_dir'],
        settings['directories']['downloaded_audio_development'])

    dir_downloaded_audio_eva = Path(
        settings['directories']['downloaded_audio_dir'],
        settings['directories']['downloaded_audio_evaluation'])

    inner_logger.info('Creating the development split data.')
    split_func(csv_development, dir_split_dev, dir_downloaded_audio_dev)
    inner_logger.info('Done.')

    nb_files_audio, nb_files_data = check_amount_of_files(
        dir_root.joinpath(dir_downloaded_audio_dev), dir_split_dev)

    inner_logger.info('Amount of development audio files: {}'.format(nb_files_audio))
    inner_logger.info('Amount of development data files: {}'.format(nb_files_data))
    inner_logger.info('Amount of development data files per audio: {}'.format(nb_files_data / nb_files_audio))

    inner_logger.info('Creating the evaluation split data.')
    split_func(csv_evaluation, dir_split_eva, dir_downloaded_audio_eva)
    inner_logger.info('Done.')

    nb_files_audio, nb_files_data = check_amount_of_files(
        dir_root.joinpath(dir_downloaded_audio_eva), dir_split_eva)

    inner_logger.info('Amount of evaluation audio files: {}'.format(nb_files_audio))
    inner_logger.info('Amount of evaluation data files: {}'.format(nb_files_data))
    inner_logger.info('Amount of evaluation data files per audio: {}'.format(nb_files_data/nb_files_audio))

    inner_logger.info('Checking the development split.')
    check_data_for_split(
        dir_audio=dir_root.joinpath(dir_downloaded_audio_dev),
        dir_data=Path(settings['dir_output'], settings['dir_data_development']),
        dir_root=dir_root, csv_split=csv_development,
        settings_ann=settings['annotations'], settings_audio=settings['audio'],
        settings_cntr=settings['counters'])
    inner_logger.info('Done.')

    inner_logger.info('Checking the evaluation split.')
    check_data_for_split(
        dir_audio=dir_root.joinpath(dir_downloaded_audio_dev),
        dir_data=Path(settings['dir_output'], settings['dir_data_evaluation']),
        dir_root=dir_root, csv_split=csv_development,
        settings_ann=settings['annotations'], settings_audio=settings['audio'],
        settings_cntr=settings['counters'])
    inner_logger.info('Done.')


def main():
    logger.remove()
    logger.add(stdout, format='{level} | [{time:HH:mm:ss}] {name} -- {message}.',
               level='INFO', filter=lambda record: record['extra']['indent'] == 1)
    logger.add(stdout, format='  {level} | [{time:HH:mm:ss}] {name} -- {message}.',
               level='INFO', filter=lambda record: record['extra']['indent'] == 2)
    main_logger = logger.bind(indent=1)

    args = get_argument_parser().parse_args()

    main_logger.info('Doing only dataset creation')

    if not args.verbose:
        main_logger.info('Verbose if off. Not logging messages')
        logger.disable('__main__')
        logger.disable('processes')

    main_logger.info(datetime.now().strftime('%Y-%m-%d %H:%M'))

    main_logger.info('Loading settings')
    settings = load_settings_file(args.config_file)
    main_logger.info('Settings loaded')

    main_logger.info('Starting Clotho dataset creation')
    create_dataset(settings)
    main_logger.info('Dataset created')


if __name__ == '__main__':
    main()

# EOF
