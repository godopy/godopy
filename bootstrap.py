#!/usr/bin/env python

import os
import sys
import glob
import shutil
import subprocess

if not hasattr(sys, 'version_info') or sys.version_info < (3, 6):
    raise SystemExit("PyGodot requires Python version 3.6 or above.")

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

    exe_glob = 'godot.*.64.exe' if sys.platform == 'win32' else 'godot.*.64'
    godot_exe_list = glob.glob(os.path.join(godot_build_dir, 'bin', exe_glob))
    if not godot_exe_list:
        raise SystemExit("Can't find Godot executable.")

    godot_exe = godot_exe_list.pop()
    api_path = os.path.join(headers_dir, 'api.json')

    print('Generating GDNative API JSON…')
    subprocess.run([godot_exe, '--gdnative-generate-json-api', api_path], check=True)

    with open(os.path.join(headers_dir, '__init__.py'), 'w', encoding='utf-8'):
        pass  # Empty file


if __name__ == '__main__':
    from binding_generator import generate_bindings
    from internal_python_build import build_python

    if not os.path.exists(prefix) and sys.platform != 'win32':
        # Windows build would kill all other Python processes
        build_python()

    if os.path.exists(headers_dir):
        print('Removing old %r…' % headers_dir)
        shutil.rmtree(headers_dir)

    copy_headers()

    os.chdir(root_dir)
    print('Generating C++ bindings…')
    generate_bindings(os.path.join(headers_dir, 'api.json'))
