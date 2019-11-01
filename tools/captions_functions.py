#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter, defaultdict
import re
from itertools import chain
from functools import partial
from tools import csv_functions


__author__ = 'Konstantinos Drossos, Samuel Lipping -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_words_counter', 'clean_sentence', 'get_sentence_words']


def get_sentence_words(sentence, unique=False, keep_case=False,
                       remove_punctuation=True, remove_specials=True):
    """Splits input sentence into words.
    
    :param sentence: Sentence to split
    :type sentence: str
    :param unique: Returns a list of unique words.
    :type unique: bool
    :param keep_case: Keep capitals and small (True) or turn\
                      everything to small case (False)
    :type keep_case: bool
    :param remove_punctuation: Remove punctuation from sentence?
    :type remove_punctuation: bool
    :param remove_specials: Remove special tokens?
    :type remove_specials: bool
    :return: Sentence words
    :rtype: list[str]
    """
    words = clean_sentence(
        sentence, keep_case=keep_case,
        remove_punctuation=remove_punctuation,
        remove_specials=remove_specials).strip().split()

    if unique:
        words = list(set(words))

    return words


def clean_sentence(sentence, keep_case=False, remove_punctuation=True, remove_specials=True):
    """Cleans a sentence.

    :param sentence: Sentence to be clean.
    :type sentence: str
    :param keep_case: Keep capitals and small (True) or turn\
                      everything to small case (False)
    :type keep_case: bool
    :param remove_punctuation: Remove punctuation from sentence?
    :type remove_punctuation: bool
    :param remove_specials: Remove special tokens?
    :type remove_specials: bool
    :return: Cleaned sentence.
    :rtype: str
    """
    the_sentence = sentence if keep_case else sentence.lower()

    if remove_specials:
        the_sentence = the_sentence.replace('<SOS> ', '').replace('<sos> ', '')
        the_sentence = the_sentence.replace(' <EOS>', '').replace(' <eos>', '')

    if remove_punctuation:
        the_sentence = re.sub('[,.!?;:\"]', ' ', the_sentence)

    return the_sentence


def get_sentence_length(sentence):
    """Gets sentence length.
    
    :param sentence: Input sentence.
    :type sentence: str
    :return: Sentence length.
    :rtype: int
    """
    return len(get_sentence_words(sentence))


def get_words_counter(captions, use_unique=False, keep_case=False,
                      remove_punctuation=True, remove_specials=True):
    """Creates a Counter object from the\
    words in the captions.

    :param captions: The captions.
    :type captions: list[str]|iterable
    :param use_unique: Use unique only words from the captions?
    :type use_unique: bool
    :param keep_case: Keep capitals and small (True) or turn\
                      everything to small case (False)
    :type keep_case: bool
    :param remove_punctuation: Remove punctuation from captions?
    :type remove_punctuation: bool
    :param remove_specials: Remove special tokens?
    :type remove_specials: bool
    :return: Counter object from\
             the words in the captions.
    :rtype: collections.Counter
    """
    partial_func = partial(
        get_sentence_words,
        unique=use_unique, keep_case=keep_case,
        remove_punctuation=remove_punctuation,
        remove_specials=remove_specials
    )
    return Counter(chain.from_iterable(map(partial_func, captions)))


def group_csv_and_get_words_counter(csv_contents, split_by, captions_fields,
                                    group_by=None, use_unique=False):
    """Creates a Counter object from the\
    words in the captions of a CSV file.

    :param csv_contents: The CSV file contents.
    :type csv_contents: list[collections.OrderedDict]
    :param split_by: Delimiter for splitting words on\
                     captions.
    :type split_by: str
    :param captions_fields: The fields of captions.
    :type captions_fields: list[str]
    :param group_by: Group the captions by a field.
    :type group_by: str
    :param use_unique: Use unique only words at the grouped captions?
    :type use_unique: bool
    :return: A Counter object from\
             the words in the captions.
    :rtype: collections.Counter
    """
    group_csv = csv_functions.group_according_to_field(
        csv_content=csv_contents,
        the_field=group_by
    )

    dict_k = list(group_csv)
    words_counter = get_words_counter(
        get_all_captions(group_csv.get(dict_k[0]), captions_fields),
        split_by=split_by, use_unique=use_unique
    )

    [words_counter.update(
        get_words_counter(
            get_all_captions(group_csv.get(dict_k[i]), captions_fields),
            split_by=split_by,
            use_unique=use_unique
        )
    ) for i in range(1, len(dict_k))]

    return words_counter


def get_all_captions(csv_contents, captions_fields):
    """Retrieves all captions.

    :param csv_contents: The contents of the CSV.
    :type csv_contents: list[collections.OrderedDict]|iterator[collections.OrderedDict]
    :param captions_fields: The name of the columns that\
                             have the captions.
    :type captions_fields: list[str]
    :return: All the captions from the CSV contents.
    :rtype: list[str]
    """
    return [
        entry[captions_field]
        for entry in csv_contents
        for captions_field in captions_fields
    ]


def output_list_of_infrequent_words(file_name, freq, counter):
    """Finds and saves to file a list of infrequent words.

    :param file_name: The file name to be used.
    :type file_name: str
    :param freq: The maximum frequency for a word to be\
                 considered infrequent.
    :type freq: int
    :param counter: The words counter object.
    :type counter: collections.Counter
    """
    to_output = sorted(k for k, v in counter.items() if v == freq)

    with open(file_name, 'w') as f:
        f.write('\n'.join(to_output))


def get_captions_fields(s_tem, nb_cap, o_set=1):
    """Returns the fields for the captions.

    :param s_tem: The template string to be used.
    :type s_tem: str
    :param nb_cap: The amount of captions.
    :type nb_cap: int
    :param o_set: Offset ot be added at the int part\
                  of the caption.
    :type o_set: int
    :return: The fields for the captions.
    :rtype: list[str]
    """
    return ' '.join(map(
        s_tem.format,
        range(o_set, nb_cap + o_set)
    )).split()


def get_captions_with_rare_words(words_set, csv_contents, captions_fields,
                                 field_id, word_freq, split_by):
    """Returns the set of captions that have a rare word.

    :param words_set: The set of words.
    :type words_set: collections.Counter
    :param csv_contents: The contents of the CSV file.
    :type csv_contents: list[collections.OrderedDict]
    :param captions_fields: The CSV fields for the captions.
    :type captions_fields: list[str]
    :param field_id: The field to be used as an ID for \
                     captions of different sounds.
    :type field_id: str
    :param word_freq: The maximum frequency for the rare\
                      word.
    :type word_freq: int
    :param split_by: The delimiter to split the caption.
    :type split_by: str
    :return: A dictionary with keys the values of the\
             different `field_id` entries and keys a list\
             of dicts, with keys the caption field and values\
             the corresponding caption.
    :rtype: dict[str, dict[str, str]]
    """
    to_return = defaultdict(dict)

    [
        to_return[csv_entry[field_id]].update({
            caption_field: csv_entry[caption_field]
            if any([
                words_set.get(word, word_freq + 1) <= word_freq
                for word in get_sentence_words(csv_entry[caption_field])
            ]) else ''
        })
        for csv_entry in csv_contents
        for caption_field in captions_fields
    ]

    [to_return.pop(csv_entry[field_id]) for csv_entry in csv_contents if all([
        to_return[csv_entry[field_id]][caption_field] == '' for caption_field in captions_fields
    ])]

    return to_return

# EOF
