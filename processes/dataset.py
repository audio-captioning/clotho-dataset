#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import stdout
from datetime import datetime
from pathlib import Path
from itertools import chain
from collections import Counter
from functools import partial

import numpy as np
from loguru import logger

from tools.argument_parsing import get_argument_parser
from tools.aux_functions import get_annotations_files, check_amount_of_files
from tools.file_io import load_audio_file, dump_numpy_object, dump_pickle_file, \
    load_settings_file
from tools.captions_functions import get_sentence_words, clean_sentence, \
    get_words_counter

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['create_dataset']


def _create_split_data(csv_split, dir_split, dir_audio, dir_root, words_list,
                       chars_list, settings_ann, settings_audio, settings_output):
    """Creates the data for the split.

    :param csv_split: Annotations of the split.
    :type csv_split: list[collections.OrderedDict]
    :param dir_split: Directory for the split.
    :type dir_split: pathlib.Path
    :param dir_audio: Directory of the audio files for the split.
    :type dir_audio: pathlib.Path
    :param dir_root: Root directory of data.
    :type dir_root: pathlib.Path
    :param words_list: List of the words.
    :type words_list: list[str]
    :param chars_list: List of the characters.
    :type chars_list: list[str]
    :param settings_ann: Settings for the annotations.
    :type settings_ann: dict
    :param settings_audio: Settings for the audio.
    :type settings_audio: dict
    :param settings_output: Settings for the output files.
    :type settings_output: dict
    """
    # Make sure that the directory exists
    dir_split.mkdir(parents=True, exist_ok=True)

    # For each sound:
    for csv_entry in csv_split:
        file_name_audio = csv_entry[settings_ann['audio_file_column']]

        audio = load_audio_file(
            audio_file=str(dir_root.joinpath(dir_audio, file_name_audio)),
            sr=int(settings_audio['sr']), mono=settings_audio['to_mono'])

        captions_fields = [settings_ann['captions_fields_prefix'].format(i)
                           for i in range(1, int(settings_ann['nb_captions']) + 1)]

        for caption_ind, caption_field in enumerate(captions_fields):
            caption = csv_entry[caption_field]

            words_caption = get_sentence_words(
                caption, unique=settings_ann['use_unique_words_per_caption'],
                keep_case=settings_ann['keep_case'],
                remove_punctuation=settings_ann['remove_punctuation_words'],
                remove_specials=not settings_ann['use_special_tokens']
            )

            chars_caption = list(chain.from_iterable(
                clean_sentence(
                    caption,
                    keep_case=settings_ann['keep_case'],
                    remove_punctuation=settings_ann['remove_punctuation_words'],
                    remove_specials=True)))

            if settings_ann['use_special_tokens']:
                chars_caption.insert(0, '<sos>')
                chars_caption.append('<eos>')

            indices_words = [words_list.index(word) for word in words_caption]
            indices_chars = [chars_list.index(char) for char in chars_caption]

            #   create the numpy object with all elements
            np_rec_array = np.rec.array(np.array(
                (file_name_audio, audio, caption, caption_ind,
                 np.array(indices_words), np.array(indices_chars)),
                dtype=[
                    ('file_name', 'U{}'.format(len(file_name_audio))),
                    ('audio_data', np.dtype(object)),
                    ('caption', 'U{}'.format(len(caption))),
                    ('caption_ind', 'i4'),
                    ('words_ind', np.dtype(object)),
                    ('chars_ind', np.dtype(object))
                ]
            ))

            #   save the numpy object to disk
            dump_numpy_object(
                np_obj=np_rec_array,
                file_name=str(dir_split.joinpath(
                    settings_output['file_name_template'].format(
                        audio_file_name=file_name_audio, caption_index=caption_ind))))


def _create_lists_and_frequencies(captions, dir_root, settings_ann, settings_cntr):
    """Creates the pickle files with words, characters, and their frequencies.

    :param captions: Captions to be used (development captions are suggested).
    :type captions: list[str]
    :param dir_root: Root directory of data.
    :type dir_root: pathlib.Path
    :param settings_ann: Settings for annotations.
    :type settings_ann: dict
    :param settings_cntr: Settings for pickle files.
    :type settings_cntr: dict
    :return: Words and characters list.
    :rtype: list[str], list[str]
    """
    # Get words counter
    counter_words = get_words_counter(
        captions=captions,
        use_unique=settings_ann['use_unique_words_per_caption'],
        keep_case=settings_ann['keep_case'],
        remove_punctuation=settings_ann['remove_punctuation_words'],
        remove_specials=not settings_ann['use_special_tokens']
    )

    # Get words and frequencies
    words_list, frequencies_words = list(counter_words.keys()), list(counter_words.values())

    # Get characters and frequencies
    cleaned_captions = [clean_sentence(
        sentence, keep_case=settings_ann['keep_case'],
        remove_punctuation=settings_ann['remove_punctuation_words'],
        remove_specials=True) for sentence in captions]

    characters_all = list(chain.from_iterable(cleaned_captions))
    counter_characters = Counter(characters_all)

    # Add special characters
    if settings_ann['use_special_tokens']:
        counter_characters.update(['<sos>'] * len(cleaned_captions))
        counter_characters.update(['<eos>'] * len(cleaned_captions))

    chars_list, frequencies_chars = list(counter_characters.keys()), list(counter_characters.values())

    # Save to disk
    obj_list = [words_list, frequencies_words, chars_list, frequencies_chars]
    obj_f_names = [
        settings_cntr['words_list_file_name'],
        settings_cntr['words_counter_file_name'],
        settings_cntr['characters_list_file_name'],
        settings_cntr['characters_frequencies_file_name']
    ]

    [dump_pickle_file(obj=obj, file_name=dir_root.joinpath(obj_f_name))
     for obj, obj_f_name in zip(obj_list, obj_f_names)]

    return words_list, chars_list


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
    words_list, chars_list = _create_lists_and_frequencies(
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
        _create_split_data,
        dir_root=dir_root,
        words_list=words_list, chars_list=chars_list,
        settings_ann=settings['annotations'],
        settings_audio=settings['audio'],
        settings_output=settings['output_files']
    )

    inner_logger.info('Creating the development split data.')
    split_func(csv_development, dir_split_dev, Path(
            settings['directories']['downloaded_audio_dir'],
            settings['directories']['downloaded_audio_development'],
        ))
    inner_logger.info('Done.')

    nb_files_audio, nb_files_data = check_amount_of_files(dir_root.joinpath(
        settings['directories']['downloaded_audio_dir'],
        settings['directories']['downloaded_audio_development'],
    ), dir_split_dev)
    inner_logger.info('Amount of development audio files: {}'.format(nb_files_audio))
    inner_logger.info('Amount of development data files: {}'.format(nb_files_data))
    inner_logger.info('Amount of development data files per audio: {}'.format(nb_files_data / nb_files_audio))

    inner_logger.info('Creating the evaluation split data.')
    split_func(csv_evaluation, dir_split_eva, Path(
        settings['directories']['downloaded_audio_dir'],
        settings['directories']['downloaded_audio_evaluation'],
    ))
    inner_logger.info('Done.')

    nb_files_audio, nb_files_data = check_amount_of_files(dir_root.joinpath(
        settings['directories']['downloaded_audio_dir'],
        settings['directories']['downloaded_audio_evaluation'],
    ), dir_split_eva)

    inner_logger.info('Amount of evaluation audio files: {}'.format(
        nb_files_audio))
    inner_logger.info('Amount of evaluation data files: {}'.format(
        nb_files_data))
    inner_logger.info('Amount of evaluation data files per audio: {}'.format(
        nb_files_data/nb_files_audio))


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
