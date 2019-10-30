#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['YAMLLoader']


class YAMLLoader(yaml.SafeLoader):
    """Custom YAML loader for adding the functionality\
    of including one YAML file inside another.

    Code after: https://stackoverflow.com/a/9577670
    """

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]
        super(YAMLLoader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, YAMLLoader)


YAMLLoader.add_constructor('!include', YAMLLoader.include)

# EOF
