#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableSequence, MutableMapping, \
    Tuple, List, Any
from functools import partial
from itertools import chain
from collections import Counter
from pathlib import Path

from tools.csv_functions import read_csv_file
from tools.captions_functions import clean_sentence, \
    get_words_counter
from tools.file_io import load_pickle_file, \
    dump_pickle_file
from tools.multi_processing import check_data_for_split_sub, \
    create_split_data_sub, do_sub_function

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_annotations_files',
           'check_data_for_split',
           'create_split_data',
           'create_lists_and_frequencies']


def check_data_for_split(dir_audio: Path,
                         dir_data: Path,
                         dir_root: Path,
                         csv_split: MutableSequence[MutableMapping[str, str]],
                         settings_ann: MutableMapping[str, Any],
                         settings_audio: MutableMapping[str, Any],
                         settings_cntr: MutableMapping[str, Any],
                         nb_workers: int) \
        -> None:
    """Goes through all audio files and checks the created data.

    Gets each audio file and checks if there are associated data. If there are,\
    checks the validity of the raw audio data and the validity of the captions,\
    words, and characters.

    :param dir_audio: Directory with the audio files.
    :type dir_audio: pathlib.Path
    :param dir_data: Directory with the data to be checked.
    :type dir_data: pathlib.Path
    :param dir_root: Root directory.
    :type dir_root: pathlib.Path
    :param csv_split: CSV entries for the data/
    :type csv_split: list[collections.OrderedDict]
    :param settings_ann: Settings for annotations.
    :type settings_ann: dict
    :param settings_audio: Settings for audio.
    :type settings_audio: dict
    :param settings_cntr: Settings for counters.
    :type settings_cntr: dict
    :param nb_workers: Amount of workers to use. If < 1, then no\
                    multi-process is happening.
    :type nb_workers: int
    """
    # Load the words and characters lists
    words_list = load_pickle_file(dir_root.joinpath(
        settings_cntr['words_list_file_name']))
    chars_list = load_pickle_file(dir_root.joinpath(
        settings_cntr['characters_list_file_name']))

    sub_f = partial(
        check_data_for_split_sub,
        dir_root=dir_root,
        dir_data=dir_data,
        dir_audio=dir_audio,
        words_list=words_list,
        chars_list=chars_list,
        settings_ann=settings_ann,
        settings_audio=settings_audio)

    do_sub_function(sub_f=sub_f, it=csv_split,
                    nb_workers=nb_workers)


def create_lists_and_frequencies(captions: MutableSequence[str],
                                 dir_root: Path,
                                 settings_ann: MutableMapping[str, Any],
                                 settings_cntr: MutableMapping[str, Any]) -> \
        Tuple[List[str], List[str]]:
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
        remove_specials=not settings_ann['use_special_tokens'])

    # Get words and frequencies
    words_list, frequencies_words = list(counter_words.keys()), \
                                    list(counter_words.values())

    # Get characters and frequencies
    cleaned_captions = [clean_sentence(
        sentence, keep_case=settings_ann['keep_case'],
        remove_punctuation=settings_ann['remove_punctuation_chars'],
        remove_specials=True) for sentence in captions]

    characters_all = list(chain.from_iterable(cleaned_captions))
    counter_characters = Counter(characters_all)

    # Add special characters
    if settings_ann['use_special_tokens']:
        counter_characters.update(['<sos>'] * len(cleaned_captions))
        counter_characters.update(['<eos>'] * len(cleaned_captions))

    chars_list, frequencies_chars = list(counter_characters.keys()), \
                                    list(counter_characters.values())

    # Save to disk
    obj_list = [words_list, frequencies_words, chars_list, frequencies_chars]
    obj_f_names = [
        settings_cntr['words_list_file_name'],
        settings_cntr['words_counter_file_name'],
        settings_cntr['characters_list_file_name'],
        settings_cntr['characters_frequencies_file_name']]

    [dump_pickle_file(obj=obj, file_name=dir_root.joinpath(obj_f_name))
     for obj, obj_f_name in zip(obj_list, obj_f_names)]

    return words_list, chars_list


def create_split_data(csv_split: MutableSequence[MutableMapping[str, str]],
                      dir_split: Path,
                      dir_audio: Path,
                      dir_root: Path,
                      words_list: MutableSequence[str],
                      chars_list: MutableSequence[str],
                      settings_ann: MutableMapping[str, Any],
                      settings_audio: MutableMapping[str, Any],
                      settings_output: MutableMapping[str, Any],
                      nb_workers: int) \
        -> None:
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
    :param nb_workers: Amount of workers to use. If < 1, then no\
                    multi-process is happening.
    :type nb_workers: int
    """
    # Make sure that the directory exists
    dir_split.mkdir(parents=True, exist_ok=True)

    captions_fields = [settings_ann['captions_fields_prefix'].format(i)
                       for i in range(1, int(settings_ann['nb_captions']) + 1)]

    sub_f = partial(
        create_split_data_sub,
        captions_fields=captions_fields,
        dir_root=dir_root,
        dir_split=dir_split,
        dir_audio=dir_audio,
        words_list=words_list,
        chars_list=chars_list,
        settings_output=settings_output,
        settings_ann=settings_ann,
        settings_audio=settings_audio)

    do_sub_function(sub_f=sub_f, it=csv_split, nb_workers=nb_workers)


def get_annotations_files(settings_ann: MutableMapping[str, Any],
                          dir_ann: Path) -> \
        Tuple[List[MutableMapping[str, Any]], List[MutableMapping[str, Any]]]:
    """Reads, process (if necessary), and returns tha annotations files.

    :param settings_ann: Settings to be used.
    :type settings_ann: dict
    :param dir_ann: Directory of the annotations files.
    :type dir_ann: pathlib.Path
    :return: Development and evaluation annotations files.
    :rtype: list[collections.OrderedDict], list[collections.OrderedDict]
    """
    field_caption = settings_ann['captions_fields_prefix']
    csv_development = read_csv_file(
        file_name=settings_ann['development_file'],
        base_dir=dir_ann)
    csv_evaluation = read_csv_file(
        file_name=settings_ann['evaluation_file'],
        base_dir=dir_ann)

    caption_fields = [field_caption.format(c_ind) for c_ind in range(1, 6)]

    for csv_entry in chain(csv_development, csv_evaluation):
        # Clean sentence to remove any spaces before punctuations.

        captions = [clean_sentence(
            csv_entry.get(caption_field),
            keep_case=True,
            remove_punctuation=False,
            remove_specials=False)
            for caption_field in caption_fields]

        if settings_ann['use_special_tokens']:
            captions = ['<SOS> {} <EOS>'.format(caption)
                        for caption in captions]

        [csv_entry.update({caption_field: caption})
         for caption_field, caption in zip(caption_fields, captions)]

    return csv_development, csv_evaluation

# EOF
