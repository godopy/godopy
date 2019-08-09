#!/usr/bin/env python3

import os
import sys
import subprocess


def build_python():
    root_dir = os.path.abspath(os.path.dirname(__file__))

    prefix = os.path.join(root_dir, 'buildenv')
    cwd = os.path.abspath(os.getcwd())
    python_path = os.path.join(root_dir, 'deps', 'python')

    print('*** Building in %r ***' % python_path)
    os.chdir(python_path)

    if sys.platform == 'win32':
        os.chdir('.\\PCBuild')
        print(os.getcwd())
        commands = [
            'build.bat -p x64 -c Debug --no-tkinter -t Build'
        ]
        subprocess.run(commands[0].split())
        os.chdir(cwd)
        sys.exit(0)

    commands = [
        f'./configure --prefix={prefix}'
        # ' --enable-optimizations'
        ' --enable-loadable-sqlite-extensions'
        ' --enable-shared',
        'make -j%d' % max(os.cpu_count() - 1, 1),
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
        print()
        print('***', command, '***')
        subprocess.run(command.split(), check=True)

    os.chdir(cwd)


if __name__ == '__main__':
    build_python()
