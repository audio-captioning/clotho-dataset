#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, List, MutableSequence
from re import sub as re_sub
from collections import Counter
from itertools import chain
from functools import partial


__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_words_counter', 'clean_sentence', 'get_sentence_words']


def get_sentence_words(sentence: str,
                       unique: Optional[bool] = False,
                       keep_case: Optional[bool] = False,
                       remove_punctuation: Optional[bool] = True,
                       remove_specials: Optional[bool] = True) -> List[str]:
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


def clean_sentence(sentence: str,
                   keep_case: Optional[bool] = False,
                   remove_punctuation: Optional[bool] = True,
                   remove_specials: Optional[bool] = True) -> str:
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

    # Remove any forgotten space before punctuation and double space.
    the_sentence = re_sub(r'\s([,.!?;:"](?:\s|$))', r'\1', the_sentence).replace('  ', ' ')

    if remove_specials:
        the_sentence = the_sentence.replace('<SOS> ', '').replace('<sos> ', '')
        the_sentence = the_sentence.replace(' <EOS>', '').replace(' <eos>', '')

    if remove_punctuation:
        the_sentence = re_sub('[,.!?;:\"]', '', the_sentence)

    return the_sentence


def get_words_counter(captions: MutableSequence[str],
                      use_unique: Optional[bool] = False,
                      keep_case: Optional[bool] = False,
                      remove_punctuation: Optional[bool] = True,
                      remove_specials: Optional[bool] = True) -> Counter:
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

# EOF
