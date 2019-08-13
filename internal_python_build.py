#!/usr/bin/env python3

import os
import sys
import subprocess

DEFAULT_TARGET = 'debug'


def build_python():
    root_dir = os.path.abspath(os.path.dirname(__file__))

    prefix = os.path.join(root_dir, 'buildenv')
    cwd = os.path.abspath(os.getcwd())
    python_path = os.path.join(root_dir, 'deps', 'python')

    if 'VIRTUAL_ENV' in os.environ:
        raise SystemExit("Please deactivate virtualenv")

    os.chdir(python_path)
    print("Building internal Python interpreter in", repr(python_path), '...')

    release_build = DEFAULT_TARGET == 'release'

    for arg in sys.argv:
        if 'release' in arg.lower():
            release_build = True
            break
        elif 'debug' in arg.lower():
            release_build = False
            break

    if sys.platform == 'win32':
        os.chdir('.\\PCBuild')
        print(os.getcwd())
        commands = [
            'build.bat -p x64 -c {0} --no-tkinter -t Build'
            .format('Release' if release_build else 'Debug')
        ]
        subprocess.run(commands[0].split())
        os.chdir(cwd)
        sys.exit(0)

    commands = [
        './configure --prefix={0} --quiet {1}'
        ' --enable-loadable-sqlite-extensions'
        ' --enable-shared'.format(prefix, '--enable-optimizations' if release_build else '--with-pydebug'),
        'make -j{0}'.format(max(os.cpu_count() - 1, 1)),
        'make install'
    ]

    if sys.platform == 'darwin':
        os.environ['OPENSSL_LIBS'] = '-lssl -lcrypto'

        os.environ['LDFLAGS'] = (
            '-L/usr/local/opt/openssl/lib '
            '-L/usr/local/opt/sqlite/lib '
        )

        os.environ['CPPFLAGS'] = (
            '-I/usr/local/opt/openssl/include '
            '-I/usr/local/opt/sqlite/include '
        )
    else:
        pass
        # os.environ['CFLAGS'] = '-fPIC'
        # os.environ['CC'] = 'clang'

    for command in commands:
        print(command)
        subprocess.run(command.split(), check=True)

    os.chdir(cwd)


if __name__ == '__main__':
    build_python()
