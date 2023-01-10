import argparse
import os
import yaml


class Dict2Object:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


def load_config(args=None):
    assert os.path.exists(args.config_file), "Invalid configs file {0}".format(args.config_file)
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    config = Dict2Object(config)
    return config


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to configs file")
    args = parser.parse_args()
    return args


class InitWithDict(object):
    """
    Base class which can be initialized by reading properties from dict
    """

    def __init__(self, init=None):
        """
        :param init:
        """

        if init:
            for key, value in init.iteritems():
                setattr(self, key, value)
