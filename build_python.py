import os
import sys
import subprocess


def build_python():
    root_dir = os.path.abspath(os.path.dirname(__file__))
    headers_dir = os.path.join(root_dir, 'godot_headers')

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
        ' --disable-shared',
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
        os.environ['CFLAGS'] = '-fPIC'
        os.environ['CC'] = 'clang'

    for command in commands:
        print()
        print('***', command, '***')
        subprocess.run(command.split(), check=True)

    # os.chdir(root_dir)
    # install_self = 'buildenv/bin/pip3 install -e .'
    # print()
    # print('\t***', install_self, '***')
    # subprocess.run(install_self.split())

    os.chdir(cwd)


if __name__ == '__main__':
    build_python()
