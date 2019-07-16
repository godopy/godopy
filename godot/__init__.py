import os
import sys
import subprocess

ROOTDIR = os.path.abspath((os.path.join(os.path.dirname(__file__), '..', '..', '..')))
if (os.path.isdir(ROOTDIR) and ROOTDIR.endswith('pyres')):
    sys.path.append(ROOTDIR)

try:
    # TODO: Rename and rearrange Cython modules
    import Godot as gdnative, Bindings as nodes
    from Godot import _pyprint as print, _gdprint as printf
except ImportError:
    # not inside Godot
    pass

# TODO: Write PyGodot version to a C header file

VERSION = (0, 0, 1, 'alpha', 0)

def githead_sha():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(repo_dir):
        return '<devpkg>'

    git_revparse = subprocess.Popen(
        'git rev-parse HEAD',
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        shell=True, cwd=repo_dir, universal_newlines=True
    )
    sha = git_revparse.communicate()[0]
    return sha[:8]

def get_version():
    main = '.'.join(str(x) for x in VERSION[:3])

    if VERSION[3] == 'final':
        return main

    mapping = {'alpha': 'a', 'beta': 'b', 'rc': 'rc'}
    sub = mapping[VERSION[3]]
    sha = githead_sha()

    if VERSION[4] == 0 and sha:
        sub += f'.dev.custom_build.{sha}'
    else:
        sub += str(VERSION[4])

    return main + sub

__version__ = get_version()
