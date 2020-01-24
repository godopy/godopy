import os
import sys
import importlib


def _nativescript_init():
    sys.path.insert(0, os.path.realpath(os.environ['SCRIPT_PATH']))
    importlib.import_module(os.environ['GODOPY_MAIN_MODULE'])._init()
