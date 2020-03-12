#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableMapping, MutableSequence, Union
from multiprocessing import Pool
from pathlib import Path
from itertools import chain

from numpy import rec, dtype, array

from tools.file_io import dump_numpy_object, \
    load_numpy_object, load_audio_file
from tools.captions_functions import clean_sentence, \
    get_sentence_words

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['check_data_for_split_sub',
           'create_split_data_sub',
           'do_sub_function',
           'extract_features_sub']


def check_data_for_split_sub(csv_entry: MutableMapping[str, str],
                             dir_root: Path,
                             dir_data: Path,
                             dir_audio: Path,
                             words_list: MutableSequence[str],
                             chars_list: MutableSequence[str],
                             settings_ann: MutableMapping[str, Union[str, int, bool, MutableMapping]],
                             settings_audio: MutableMapping[str, Union[str, int, bool, MutableMapping]]) \
        -> None:
    """Sub function for data checking.

    :param csv_entry: CSV entry for each sound.
    :type csv_entry: dict[str, str]
    :param dir_root: Root directory for dataset.
    :type dir_root: pathlib.Path
    :param dir_data: Directory of dataset.
    :type dir_data: pathlib.Path
    :param dir_audio: Directory of the audio files for the split.
    :type dir_audio: pathlib.Path
    :param words_list: List of the words.
    :type words_list: list[str]
    :param chars_list: List of the characters.
    :type chars_list: list[str]
    :param settings_ann: Settings for the annotations.
    :type settings_ann: dict
    :param settings_audio: Settings for the audio.
    :type settings_audio: dict
    """
    # Get audio file name
    file_name_audio = Path(csv_entry[settings_ann['audio_file_column']])

    # Check if the audio file existed originally
    if not dir_audio.joinpath(file_name_audio).exists():
        raise FileExistsError('Audio file {f_name_audio} not exists in {d_audio}'.format(
            f_name_audio=file_name_audio, d_audio=dir_audio))

    # Flag for checking if there are data files for the audio file
    audio_has_data_files = False

    # Get the original audio data
    data_audio_original = load_audio_file(
        audio_file=str(dir_audio.joinpath(file_name_audio)),
        sr=int(settings_audio['sr']), mono=settings_audio['to_mono'])

    for data_file in dir_root.joinpath(dir_data).iterdir():
        # Get the stem of the audio file name
        f_stem = str(data_file).split('file_')[-1].split('.wav_')[0]

        if f_stem == file_name_audio.stem:
            audio_has_data_files = True
            # Get the numpy record array
            data_array = load_numpy_object(data_file)

            # Get the audio data from the numpy record array
            data_audio_rec_array = data_array['audio_data'].item()

            # Compare the lengths
            if len(data_audio_rec_array) != len(data_audio_original):
                raise ValueError(
                    'File {f_audio} was not saved successfully to the numpy '
                    'object {f_np}.'.format(f_audio=file_name_audio, f_np=data_file))

            # Check all elements, one to one
            if not all([data_audio_original[i] == data_audio_rec_array[i]
                        for i in range(len(data_audio_original))]):
                raise ValueError('Numpy object {} has wrong audio data.'.format(data_file))

            # Get the original caption
            caption_index = data_array['caption_ind'].item()

            # Clean it to remove any spaces before punctuation.
            original_caption = clean_sentence(
                sentence=csv_entry[settings_ann['captions_fields_prefix'].format(caption_index + 1)],
                keep_case=True, remove_punctuation=False,
                remove_specials=not settings_ann['use_special_tokens'])

            # Check with the file caption
            caption_data_array = clean_sentence(
                sentence=data_array['caption'].item(), keep_case=True,
                remove_punctuation=False,
                remove_specials=not settings_ann['use_special_tokens'])

            if not original_caption == caption_data_array:
                raise ValueError('Numpy object {} has wrong caption.'.format(data_file))

            # Since caption in the file is OK, we can use it instead of
            # the original, because it already has the special tokens.
            caption_data_array = clean_sentence(
                sentence=data_array['caption'].item(),
                keep_case=settings_ann['keep_case'],
                remove_punctuation=settings_ann['remove_punctuation_words'],
                remove_specials=not settings_ann['use_special_tokens'])

            # Check with the indices of words
            words_indices = data_array['words_ind'].item()
            caption_form_words = ' '.join([words_list[i] for i in words_indices])

            if not caption_data_array == caption_form_words:
                raise ValueError('Numpy object {} has wrong words indices.'.format(data_file))

            # Check with the indices of characters
            caption_from_chars = ''.join([chars_list[i] for i in data_array['chars_ind'].item()])

            caption_data_array = clean_sentence(
                sentence=data_array['caption'].item(),
                keep_case=settings_ann['keep_case'],
                remove_punctuation=settings_ann['remove_punctuation_chars'],
                remove_specials=not settings_ann['use_special_tokens'])

            if not caption_data_array == caption_from_chars:
                raise ValueError('Numpy object {} has wrong characters '
                                 'indices.'.format(data_file))

    if not audio_has_data_files:
        raise FileExistsError('Audio file {} has no associated data.'.format(
            file_name_audio))


def create_split_data_sub(csv_entry: MutableMapping[str, str],
                          captions_fields: MutableSequence[str],
                          dir_root: Path,
                          dir_split: Path,
                          dir_audio: Path,
                          words_list: MutableSequence[str],
                          chars_list: MutableSequence[str],
                          settings_output: MutableMapping[str, Union[str, int, bool, MutableMapping]],
                          settings_ann: MutableMapping[str, Union[str, int, bool, MutableMapping]],
                          settings_audio: MutableMapping[str, Union[str, int, bool, MutableMapping]]) \
        -> None:
    """Sub function for data split creation.

    :param csv_entry: CSV entry for each sound.
    :type csv_entry: dict[str, str]
    :param captions_fields: Names of the columns for captions.
    :type captions_fields: list[str]
    :param dir_root: Root directory for dataset.
    :type dir_root: pathlib.Path
    :param dir_split: Directory for the split.
    :type dir_split: pathlib.Path
    :param dir_audio: Directory of the audio files for the split.
    :type dir_audio: pathlib.Path
    :param words_list: List of the words.
    :type words_list: list[str]
    :param chars_list: List of the characters.
    :type chars_list: list[str]
    :param settings_output: Settings for the output files.
    :type settings_output: dict
    :param settings_ann: Settings for the annotations.
    :type settings_ann: dict
    :param settings_audio: Settings for the audio.
    :type settings_audio: dict
    """
    f_name_audio = csv_entry[settings_ann['audio_file_column']]

    audio = load_audio_file(
        audio_file=str(dir_root.joinpath(dir_audio, f_name_audio)),
        sr=int(settings_audio['sr']),
        mono=settings_audio['to_mono'])

    for caption_ind, caption_field in enumerate(captions_fields):
        caption = csv_entry[caption_field]

        words_caption = get_sentence_words(
            caption,
            unique=settings_ann['use_unique_words_per_caption'],
            keep_case=settings_ann['keep_case'],
            remove_punctuation=settings_ann['remove_punctuation_words'],
            remove_specials=not settings_ann['use_special_tokens'])

        chars_caption = list(chain.from_iterable(
            clean_sentence(
                caption,
                keep_case=settings_ann['keep_case'],
                remove_punctuation=settings_ann['remove_punctuation_chars'],
                remove_specials=True)))

        if settings_ann['use_special_tokens']:
            chars_caption.insert(0, ' ')
            chars_caption.insert(0, '<sos>')
            chars_caption.append(' ')
            chars_caption.append('<eos>')

        indices_words = [words_list.index(word) for word in words_caption]
        indices_chars = [chars_list.index(char) for char in chars_caption]

        #   create the numpy object with all elements
        np_rec_array = rec.array(array(
            (f_name_audio, audio, caption, caption_ind,
             array(indices_words), array(indices_chars)),
            dtype=[
                ('file_name', f'U{len(f_name_audio)}'),
                ('audio_data', dtype(object)),
                ('caption', f'U{len(caption)}'),
                ('caption_ind', 'i4'),
                ('words_ind', dtype(object)),
                ('chars_ind', dtype(object))
            ]
        ))

        #   save the numpy object to disk
        dump_numpy_object(
            np_obj=np_rec_array,
            file_name=str(dir_split.joinpath(
                settings_output['file_name_template'].format(
                    audio_file_name=f_name_audio,
                    caption_index=caption_ind))))


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
