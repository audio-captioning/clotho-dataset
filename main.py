#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import stdout
from datetime import datetime

from loguru import logger

from tools.argument_parsing import get_argument_parser
from tools.file_io import load_settings_file
from processes import create_dataset, extract_features

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['main']


def main():
    logger.remove()
    logger.add(stdout, format='{level} | [{time:HH:mm:ss}] {name} -- {message}.',
               level='INFO', filter=lambda record: record['extra']['indent'] == 1)
    logger.add(stdout, format='  {level} | [{time:HH:mm:ss}] {name} -- {message}.',
               level='INFO', filter=lambda record: record['extra']['indent'] == 2)
    main_logger = logger.bind(indent=1)

    args = get_argument_parser().parse_args()

    if not args.verbose:
        main_logger.info('Verbose if off. Not logging messages')
        logger.disable('__main__')
        logger.disable('processes')

    main_logger.info(datetime.now().strftime('%Y-%m-%d %H:%M'))

    main_logger.info('Loading settings')
    settings_dataset = load_settings_file(args.config_file_dataset)
    settings_features = load_settings_file(args.config_file_features)
    main_logger.info('Settings loaded')

    if settings_dataset['workflow']['create_dataset']:
        main_logger.info('Starting Clotho dataset creation')
        create_dataset(settings_dataset)
        main_logger.info('Dataset created')

    if settings_dataset['workflow']['extract_features']:
        main_logger.info('Starting Clotho feature extraction')
        extract_features(settings_dataset, settings_features)
        main_logger.info('Features extracted')


if __name__ == '__main__':
    main()

# EOF
