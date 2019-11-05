#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_argument_parser']


def get_argument_parser():
    """Creates and returns the ArgumentParser for this project.

    :return: The argument parser.
    :rtype: argparse.ArgumentParser
    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-d', '--config-file-dataset',
                            type=str, default='dataset_creation')
    arg_parser.add_argument('-f', '--config-file-features',
                            type=str, default='feature_extraction')
    arg_parser.add_argument('--verbose', action='store_true')

    return arg_parser

# EOF
