#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import shutil
import subprocess

if not hasattr(sys, 'version_info') or sys.version_info < (3, 6):
    raise SystemExit("Python version 3.6 or above is required. Please run from Python 3.6+ virtualenv.")

try:
    import pycparser  # noqa
    import autopxd    # noqa
    import cython     # noqa
except ImportError:
    raise SystemExit("Required packages were not found. Please install them from 'tool-requirements.txt'.")

root_dir = os.path.abspath(os.path.dirname(__file__))
headers_dir = os.path.join(root_dir, 'godot_headers')
cwd = os.path.abspath(os.getcwd())

prefix = os.path.join(root_dir, 'buildenv')


def copy_headers():
    godot_build_dir = os.environ.get('GODOT_BUILD')
    if not godot_build_dir:
        raise SystemExit("'GODOT_BUILD' environment variable is required.")

    source_dir = os.path.join(godot_build_dir, 'modules', 'gdnative', 'include')
    print('Copying godot_headers from %r…' % source_dir)
    shutil.copytree(source_dir, headers_dir)

    godot_server_exe_list = crossplat_exe_glob(godot_build_dir, 'godot?server.*.64')
    godot_exe_list = crossplat_exe_glob(godot_build_dir, 'godot.*.64')

    if not godot_server_exe_list and not godot_exe_list:
        raise SystemExit("Can't find Godot executable.")

    godot_exe = godot_server_exe_list.pop() if godot_server_exe_list else godot_exe_list.pop()
    api_path = os.path.join(headers_dir, 'api.json')

    print('Found %r executable.' % godot_exe)
    print('Generating GDNative API JSON…')
    subprocess.run([godot_exe, '--gdnative-generate-json-api', api_path], check=True)

    with open(os.path.join(headers_dir, '__init__.py'), 'w', encoding='utf-8'):
        pass  # Empty file


def crossplat_exe_glob(godot_build_dir, pattern):
    if sys.platform == 'win32':
        pattern += '.exe'

    return glob.glob(os.path.join(godot_build_dir, 'bin', pattern))


if __name__ == '__main__':
    from binding_generator import generate_bindings
    # from internal_python_build import build_python

    # if not os.path.exists(prefix) and sys.platform != 'win32':
    #     # Windows build would kill all other Python processes
    #     build_python()

    if os.path.exists(headers_dir):
        print('Removing old %r…' % headers_dir)
        shutil.rmtree(headers_dir)

    copy_headers()

    os.chdir(root_dir)
    print('Generating C++ bindings…')
    generate_bindings(os.path.join(headers_dir, 'api.json'))
