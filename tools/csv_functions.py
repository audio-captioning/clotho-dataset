#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import csv
import collections
from operator import itemgetter

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['group_according_to_field', 'order_by', 'read_csv_file']


def group_according_to_field(csv_content, the_field):
    """Groups the contents of a CSV according to field.

    :param csv_content: The contents of the CSV.
    :type csv_content: list[collections.OrderedDict]
    :param the_field: The field to group the contents.
    :type the_field: str
    :return:  A dictionary with keys the different \
              values of the field and values the entries\
              of the CSV.
    :rtype: collections.defaultdict[str, list[collections.OrderedDict]]
    """
    to_return = collections.defaultdict(list)
    [to_return[i[the_field]].append(i) for i in csv_content]
    return to_return


def order_by(csv_content, the_field, reverse=False):
    """Orders the CSV contents by a specific field.

    :param csv_content: The CSV contents.
    :type csv_content: list[collections.OrderedDict]
    :param the_field: The field to be used.
    :type the_field: str
    :param reverse: Reverse ordering?
    :type the_field: bool
    :return: The ordered contents.
    :rtype: list[collections.OrderedDict]
    """
    return sorted(
        csv_content,
        key=itemgetter[the_field],
        reverse=reverse
    )


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


def write_csv_file_with_dict(file_name, dict_list, base_dir):
    """Writes an ordered dict to a CSV with DictWriter.

    :param file_name: The full file name of the CSV.
    :type file_name: str
    :param dict_list: The list of ordered dicts to write.
    :type dict_list: list[collections.OrderedDict]
    :param base_dir: The root dir of the CSV files.
    :type base_dir: str
    """
    file_path = pathlib.Path().joinpath(base_dir, file_name)
    with file_path.open(mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_list[0].keys())
        writer.writeheader()
        writer.writerows(dict_list)

# EOF
