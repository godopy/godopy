import sys
import os
import glob


def get_godot_executable(*, noserver=False):
    godot_build_dir = os.environ.get('GODOT_BUILD')
    if not godot_build_dir:
        raise SystemExit("'GODOT_BUILD' environment variable is required.")

    if noserver:
        godot_server_exe_list = []
    else:
        godot_server_exe_list = crossplat_exe_glob(godot_build_dir, 'godot?server.*.64')

    godot_exe_list = crossplat_exe_glob(godot_build_dir, 'godot.*.64')

    if not godot_server_exe_list and not godot_exe_list:
        raise SystemExit("Can't find Godot executable.")

    return godot_server_exe_list.pop() if godot_server_exe_list else godot_exe_list.pop()


def crossplat_exe_glob(godot_build_dir, pattern):
    if sys.platform == 'win32':
        pattern += '.exe'

    return glob.glob(os.path.join(godot_build_dir, 'bin', pattern))


def is_internal_path(path):
    return path.startswith('_lib') or path.startswith('src')
