#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from contextlib import ContextDecorator
from datetime import datetime

__author__ = 'Konstantinos Drossos -- TUT'
__docformat__ = 'reStructuredText'
__all__ = [
    'print_msg', 'inform_about_device', 'print_date_and_time',
    'print_processes_message', 'InformAboutProcess', 'print_yaml_settings',
    'print_training_results', 'print_evaluation_results', 'print_intro_messages',
    'surround_with_empty_lines'
]


_time_f_spec = '7.2'
_acc_f_spec = '6.2'
_loss_f_spec = '7.3'
_epoch_f_spec = '4'


def _print_empty_lines(nb_lines=1):
    """Prints empty lines.

    :param nb_lines: The amount of lines.
    :type nb_lines: int
    """
    for _ in range(nb_lines):
        print_msg('', start='', end='\n')


def print_intro_messages(device):
    """Prints initial messages.

    :param device: The device to be used.
    :type device: str
    """
    print_date_and_time()
    print_msg(' ', start='')
    inform_about_device(device)
    print_msg(' ', start='')


def print_msg(the_msg, start='-- ', end='\n', flush=True):
    """Prints a message.

    :param the_msg: The message.
    :type the_msg: str
    :param start: Starting decoration.
    :type start: str
    :param end: Ending character.
    :type end: str
    :param flush: Flush buffer now?
    :type flush: bool
    """
    print('{}{}'.format(start, the_msg), end=end, flush=flush)


def print_yaml_settings(the_settings):
    """Prints the settings in the YAML settings file.

    :param the_settings: The settings dict
    :type the_settings: dict
    """
    def _print_dict_yaml_settings(the_dict, indentation, start):
        """Prints a nested dict.

        :param the_dict: The nested dict.
        :type the_dict: dict
        :param indentation: Indentation for the printing.
        :type indentation: str
        :param start: Starting decoration.
        :type start: str
        """
        k_l_ = max(*([len(_k) for _k in the_dict.keys()] + [0]))
        for k_, v_ in the_dict.items():
            print_msg('{k_:<{k_l_:d}s}:'.format(k_=k_, k_l_=k_l_),
                      start='{}{}|-- '.format(start, indentation),
                      end=' ', flush=True)
            start = ''
            if type(v_) == dict:
                _print_dict_yaml_settings(v_, '{}{}'.format(indentation, ' ' * 5), '\n')
            elif type(v_) == list:
                print_msg(', '.join(map(str, v_)), start='')
            else:
                print_msg(v_, start='')

    try:
        print_msg('ModelDCASE description: {}'.format(the_settings['model_description']),
                  start='\n-- ')
    except KeyError:
        pass

    print_msg('Settings: ', end='\n\n')

    dict_to_print = {k__: v__ for k__, v__ in the_settings.items() if k__ != 'model_description'}
    k_len = max(*[len(k__) for k__ in dict_to_print.keys()])

    for k, v in dict_to_print.items():
        k_len = max(k_len, len(k))
        print_msg('{}:'.format(k), start=' ' * 2, end=' ')
        if type(v) == dict:
            _print_dict_yaml_settings(v, ' ' * 3, '\n')
        else:
            print_msg(v, start='')
        print_msg('', start='')


def inform_about_device(the_device):
    """Prints an informative message about the device that we are using.

    :param the_device: The device.
    :type the_device: str
    """
    print_msg('Using device: `{}`.'.format(the_device))


def print_date_and_time():
    """Prints the date and time of `now`.
    """
    print_msg(datetime.now().strftime('%Y-%m-%d %H:%M'), start='\n\n-- ', end='\n\n')


class InformAboutProcess(ContextDecorator):
    def __init__(self, starting_msg, ending_msg='done', start='-- ', end='\n'):
        """Context manager and decorator for informing about a process.

        :param starting_msg: The starting message, printed before the process starts.
        :type starting_msg: str
        :param ending_msg: The ending message, printed after process ends.
        :type ending_msg: str
        :param start: Starting decorator for the string to be printed.
        :type start: str
        :param end: Ending decorator for the string to be printed.
        :type end: str
        """
        super(InformAboutProcess, self).__init__()
        self.starting_msg = starting_msg
        self.ending_msg = ending_msg
        self.start_dec = start
        self.end_dec = end

    def __enter__(self):
        print_msg('{}... '.format(self.starting_msg), start=self.start_dec, end='')

    def __exit__(self, *exc_type):
        print_msg('{}.'.format(self.ending_msg), start='', end=self.end_dec)
        return False


def print_processes_message(workers):
    """Prints a message for how many processes are used.

    :param workers: The amount of processes.
    :type workers: int
    """
    msg_str = '| Using {} {} |'.format('single' if workers is None else workers,
                                       'processes' if workers > 1 else 'process')
    print_msg('\n'.join(['', '*' * len(msg_str), msg_str, '*' * len(msg_str)]), flush=True)


def print_training_results(epoch, training_loss, validation_loss,
                           training_metric, validation_metric,
                           time_elapsed):
    """Prints the results of the pre-training step to console.

    :param epoch: The epoch.
    :type epoch: int
    :param training_loss: The loss of the training data.
    :type training_loss: float
    :param validation_loss: The loss of the validation data.
    :type validation_loss: float | None
    :param training_metric: The metric score for the training data.
    :type training_metric: float
    :param validation_metric: The metric score for the validation data.
    :type validation_metric: float | None
    :param time_elapsed: The time elapsed for the epoch.
    :type time_elapsed: float
    """
    the_msg = \
        'Epoch:{e:{e_spec}d} | ' \
        'Loss (tr/va):{l_tr:{l_f_spec}f}/{l_va:{l_f_spec}f} | ' \
        'Metric score (tr/va):{acc_tr:{acc_f_spec}f}/{acc_va:{acc_f_spec}f} | ' \
        'Time:{t:{t_f_spec}f} sec.'.format(
            e=epoch,
            l_tr=training_loss, l_va='None' if validation_loss is None else validation_loss,
            acc_tr=training_metric, acc_va='None' if validation_metric is None else validation_metric,
            t=time_elapsed,
            l_f_spec=_loss_f_spec, acc_f_spec=_acc_f_spec, t_f_spec=_time_f_spec,
            e_spec=_epoch_f_spec)

    print_msg(the_msg, start='  -- ')


def print_evaluation_results(accuracy, time_elapsed):
    """Prints the output of the testing process.

    :param accuracy: The accuracy.
    :type accuracy: float
    :param time_elapsed: The elapsed time for the epoch.
    :type time_elapsed: float
    """
    the_msg = 'Metric:{acc:{acc_f_spec}f} | Time:{t:{t_f_spec}f}'.format(
            acc=accuracy, t=time_elapsed,
            acc_f_spec=_acc_f_spec, t_f_spec=_time_f_spec)

    print_msg(the_msg, start='  -- ')


def surround_with_empty_lines(func, nb_lines, *args, **kwargs):
    """Surrounds a function with empty lines at the std output.

    :param func: The function to be surrounded.
    :type func: callable
    :param nb_lines: Amount of lines to be used before and after.\
                     If only one number is used, then this number is\
                     used for lines before the function calling.
    :type nb_lines: (int, int)|int
    """
    if type(nb_lines) == int:
        before_lines = nb_lines
        after_lines = 1
    else:
        before_lines = nb_lines[0]
        after_lines = nb_lines[1]
    _print_empty_lines(before_lines)
    func(*args, **kwargs)
    _print_empty_lines(after_lines)

# EOF
