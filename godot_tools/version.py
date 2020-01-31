import os
import subprocess

VERSION = (0, 0, 1, 'final')


def githead_sha():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # TODO: Write PyGodot version to a C header file and retrieve SHA from the built-in modules
    if not os.path.isdir(repo_dir):
        return '<unknown-commit>'

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
    sha = VERSION[4] == 0 and githead_sha()

    if sha:
        # Comply with PEP 440
        sub += f'0.dev0+{sha}'
    else:
        sub += str(VERSION[4])

    return main + sub
