#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from itertools import chain
from collections import Counter

from tools import csv_functions, captions_functions, file_io

__author__ = 'Konstantinos Drossos'
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


def _create_split_data(dir_split, dir_root, csv_split, settings_cntr):
    pass


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
    counter_characters = Counter(list(chain.from_iterable([list(word) for word in words_list])))
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


def create_dataset(settings):
    """Creates the dataset.

    Gets the dictionary with the settings and creates
    the files of the dataset.

    :param settings: Settings to be used.
    :type settings: dict
    """
    # Get root dir
    dir_root = Path(settings['directories']['root_dir'])

    # Read the annotation files
    csv_development, csv_evaluation = _get_annotations_files(
        settings_ann=settings['annotations'],
        dir_ann=dir_root.joinpath(settings['directories']['annotations_dir']))

    # Get all captions
    captions_development = [csv_field.get('caption_{}'.format(c_ind))
                            for csv_field in csv_development
                            for c_ind in range(1, 6)]

    _create_lists_and_frequencies(
        captions=captions_development, dir_root=dir_root,
        settings_ann=settings['annotations'],
        settings_cntr=settings['counter'])

    print('OK')

    # Make the paths for all sounds

    # For each sound:
    #   read it
    #   extract features
    #   create the numpy object with all
    #     elements
    #   save the numpy object to disk
    pass


def main():
    pass


if __name__ == '__main__':
    main()

# EOF
