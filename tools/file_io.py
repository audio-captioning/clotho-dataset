#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union, Dict, Any
from pathlib import Path
import os
import pickle
import yaml

from librosa import load
import numpy as np

from tools import yaml_loader

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = [
    'dump_numpy_object', 'dump_pickle_file',
    'load_numpy_object', 'load_pickle_file',
    'load_settings_file', 'load_audio_file',
]


def dump_numpy_object(np_obj: np.ndarray,
                      file_name: str, ext: Optional[str] = '.npy',
                      replace_ext: Optional[bool] = True) -> None:
    """Dumps a numpy object to HDD.

    :param np_obj: The numpy object.
    :type np_obj: numpy.ndarray
    :param file_name: The file name to be used.
    :type file_name: str
    :param ext: The extension for the dumped object.
    :type ext: str
    :param replace_ext: Replace extension if `file_name`\
                        has a different one?
    :type replace_ext: bool
    """
    np.save('{}{}'.format(os.path.splitext(file_name)[0], ext)
            if replace_ext and (os.path.splitext(file_name)[-1] != ext
                                or os.path.splitext(file_name)[-1] == '')
            else file_name, np_obj)


def dump_pickle_file(obj: object, file_name: Union[str, Path],
                     protocol: Optional[int] = 2):
    """Dumps an object to pickle file.

    :param obj: The object to dump.
    :type obj: object | list | dict | numpy.ndarray
    :param file_name: The resulting file name.
    :type file_name: str|pathlib.Path
    :param protocol: The protocol to be used.
    :type protocol: int
    """
    str_file_name = file_name if type(file_name) == str else str(file_name)

    with open(str_file_name, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def load_audio_file(audio_file: str, sr: int, mono: bool,
                    offset: Optional[float] = 0.0,
                    duration: Optional[Union[float, None]] = None) -> np.ndarray:
    """Loads the data of an audio file.

    :param audio_file: The path of the audio file.
    :type audio_file: str
    :param sr: The sampling frequency to be used.
    :type sr: int
    :param mono: Turn to mono?
    :type mono: bool
    :param offset: Offset to be used (in seconds).
    :type offset: float
    :param duration: Duration of signal to load (in seconds).
    :type duration: float|None
    :return: The audio data.
    :rtype: numpy.ndarray
    """
    return load(path=audio_file, sr=sr, mono=mono,
                offset=offset, duration=duration)[0]


def load_numpy_object(file_name: Union[str, Path]) -> Union[np.ndarray, np.recarray]:
    """Loads and returns a numpy object.

    :param file_name: File name of the numpy object.
    :type file_name: str|pathlib.Path
    :return: Numpy object.
    :rtype: numpy.ndarray|numpy.rec.array
    """
    return np.load(str(file_name), allow_pickle=True)


def load_pickle_file(file_name: Path, encoding: Optional[str] = 'latin1') -> Any:
    """Loads a pickle file.

    :param file_name: File name (extension included).
    :type file_name: pathlib.Path
    :param encoding: Encoding of the file.
    :type encoding: str
    :return: Loaded object.
    :rtype: object
    """
    str_file_name = file_name if type(file_name) == str else str(file_name)

    with open(str_file_name, 'rb') as f:
        return pickle.load(f, encoding=encoding)


def load_settings_file(file_name: str,
                       settings_dir: Optional[Path] = Path('settings')) \
        -> Dict[str, Any]:
    """Reads and returns the contents of a YAML settings file.

    :param file_name: The name of the settings file.
    :type file_name: str
    :param settings_dir: The directory with the settings files.
    :type settings_dir: pathlib.Path|str
    :return: The contents of the YAML settings file.
    :rtype: dict
    """
    settings_dir = Path(settings_dir) \
        if type(settings_dir) == str else settings_dir
    settings_file_path = settings_dir.joinpath('{}.yaml'.format(file_name))
    return load_yaml_file(settings_file_path)


def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Reads and returns the contents of a YAML file.

    :param file_path: The path to the YAML file.
    :type file_path: pathlib.Path|str
    :return: The contents of the YAML file.
    :rtype: dict
    """
    if type(file_path) == str:
        file_path = Path(file_path)

    with file_path.open('r') as f:
        return yaml.load(f, Loader=yaml_loader.YAMLLoader)

# EOF
