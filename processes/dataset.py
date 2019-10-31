#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from itertools import chain
from collections import Counter
from functools import partial

import numpy as np
from loguru import logger

from tools import csv_functions, captions_functions, file_io

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['create_dataset']


def _get_annotations_files(settings_ann, dir_ann):
    """Reads, process (if necessary), and returns tha annotations files.

    :param settings_ann: Settings to be used.
    :type settings_ann: dict
    :param dir_ann: Directory of the annotations files.
    :type dir_ann: pathlib.Path
    :return: Development and evaluation annotations files.
    :rtype: list[collections.OrderedDict], list[collections.OrderedDict]
    """
    field_caption = settings_ann['captions_fields_prefix']
    csv_development = csv_functions.read_csv_file(
        file_name=settings_ann['development_file'],
        base_dir=dir_ann)
    csv_evaluation = csv_functions.read_csv_file(
        file_name=settings_ann['evaluation_file'],
        base_dir=dir_ann)

    if settings_ann['use_special_tokens']:
        # Update the captions with <SOS> and <EOS>
        for csv_entry in chain(csv_development, csv_evaluation):
            caption_fields = [field_caption.format(c_ind) for c_ind in range(1, 6)]
            captions = ['<SOS> {} <EOS>'.format(csv_entry.get(caption_field))
                        for caption_field in caption_fields]
            [csv_entry.update({caption_field: caption})
             for caption_field, caption in zip(caption_fields, captions)]

    return csv_development, csv_evaluation


def _create_split_data(dir_split, dir_audio, dir_root, csv_split, words_list,
                       chars_list, settings_ann, settings_audio, settings_output):
    """

    :param dir_split:
    :type dir_split: pathlib.Path
    :param dir_audio:
    :type dir_audio: pathlib.Path
    :param dir_root:
    :type dir_root: pathlib.Path
    :param csv_split:
    :type csv_split: list[collections.OrderedDict]
    :param words_list:
    :type words_list: list[str]
    :param chars_list:
    :type chars_list: list[str]
    :param settings_ann:
    :type settings_ann: dict
    :param settings_audio:
    :type settings_audio: dict
    :param settings_output:
    :type settings_output: dict
    :return:
    :rtype:
    """
    # For each sound:
    for csv_entry in csv_split:
        file_name_audio = csv_entry[settings_ann['audio_file_column']]

        audio = file_io.load_audio_file(
            audio_file=dir_root.joinpath(dir_audio).joinpath(file_name_audio),
            sr=int(settings_audio['sr']), mono=settings_audio['to_mono'])

        captions_fields = [settings_ann['captions_fields_prefix'].format(i)
                           for i in range(1, int(settings_ann['nb_captions']) + 1)]

        for caption_ind, caption_field in enumerate(captions_fields):
            caption = csv_entry[caption_field]

            words_caption = captions_functions.get_sentence_words(
                caption, unique=settings_ann['use_unique_words_per_caption'],
                keep_case=settings_ann['keep_case'],
                remove_punctuation=settings_ann['keep_punctuation']
            )

            chars_caption = list(chain.from_iterable(
                captions_functions.clean_sentence(
                    caption,
                    keep_case=settings_ann['keep_case'],
                    remove_punctuation=settings_ann['remove_punctuation'],
                    remove_specials=True)))

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
            file_io.dump_numpy_object(
                np_obj=np_rec_array,
                file_name=dir_root.joinpath(dir_split).joinpath(
                    settings_output['file_name_template'].format(
                        audio_file_name=file_name_audio, caption_index=caption_ind)))


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
    counter_words = captions_functions.get_words_counter(
        captions=captions,
        use_unique=settings_ann['use_unique_words_per_caption'],
        keep_case=settings_ann['keep_case'],
        remove_punctuation=settings_ann['keep_punctuation']
    )

    # Get words and frequencies
    words_list, frequencies_words = list(counter_words.keys()), list(counter_words.values())

    # Get characters and frequencies
    cleaned_captions = [captions_functions.clean_sentence(
        sentence, keep_case=settings_ann['keep_case'],
        remove_punctuation=settings_ann['remove_punctuation'],
        remove_specials=True) for sentence in captions]

    characters_all = list(chain.from_iterable(cleaned_captions))
    counter_characters = Counter(characters_all)
    chars_list, frequencies_chars = list(counter_characters.keys()), list(counter_characters.values())

    # Save to disk
    obj_list = [words_list, frequencies_words, chars_list, frequencies_chars]
    obj_f_names = [
        settings_cntr['words_list_file_name'],
        settings_cntr['words_counter_file_name'],
        settings_cntr['characters_list_file_name'],
        settings_cntr['characters_frequencies_file_name']
    ]

    [file_io.dump_pickle_file(obj=obj, file_name=dir_root.joinpath(obj_f_name))
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
    csv_development, csv_evaluation = _get_annotations_files(
        settings_ann=settings['annotations'],
        dir_ann=dir_root.joinpath(settings['directories']['annotations_dir']))
    inner_logger.info('Done.')

    # Get all captions
    inner_logger.info('Getting the captions.')
    captions_development = [csv_field.get('caption_{}'.format(c_ind))
                            for csv_field in csv_development
                            for c_ind in range(1, 6)]
    inner_logger.info('Done.')

    inner_logger.info('Creating and saving words and chars lists and frequencies.')
    words_list, chars_list = _create_lists_and_frequencies(
        captions=captions_development, dir_root=dir_root,
        settings_ann=settings['annotations'],
        settings_cntr=settings['counter'])
    inner_logger.info('Done.')

    dir_split_dev = dir_root.joinpath(settings['directories']['development_dir'])
    dir_split_eva = dir_root.joinpath(settings['directories']['evaluation_dir'])

    split_func = partial(
        _create_split_data, dir_split=dir_split_dev,
        dir_audio=settings['directories']['audio_dir'],
        dir_root=dir_root, csv_split=csv_development,
        words_list=words_list, chars_list=chars_list,
        settings_ann=settings['annotations'],
        settings_audio=settings['audio'],
        settings_output=settings['output_files']
    )

    inner_logger.info('Creating the development split data.')
    split_func(dir_split_dev)
    inner_logger.info('Done.')
    inner_logger.info('Creating the evaluation split data.')
    split_func(dir_split_eva)
    inner_logger.info('Done.')


def main():
    pass


if __name__ == '__main__':
    main()

# EOF
