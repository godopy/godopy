import os
import subprocess

# TODO: Write pygdnlib version to a C header file

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
