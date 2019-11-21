import collections
from recordtype import recordtype
import argparse

from config_parser.config_parser import ConfigGenerator


def parse_args_to_config(args):
    config_parser = ConfigGenerator('config/default_config.yml')
    return config_parser(args)
