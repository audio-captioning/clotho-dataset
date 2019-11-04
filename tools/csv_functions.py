#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import csv
import collections
from operator import itemgetter

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['read_csv_file']


def read_csv_file(file_name, base_dir='csv_files'):
    """Reads a CSV file.

    :param file_name: The full file name of the CSV.
    :type file_name: str
    :param base_dir: The root dir of the CSV files.
    :type base_dir: str|pathlib.Path
    :return: The contents of the CSV of the task.
    :rtype: list[collections.OrderedDict]
    """
    file_path = pathlib.Path().joinpath(base_dir, file_name)
    with file_path.open(mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        return [csv_line for csv_line in csv_reader]

# EOF
