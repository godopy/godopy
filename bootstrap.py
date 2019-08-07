import os
import sys
import subprocess


def build():
    root_dir = os.path.abspath(os.path.dirname(__file__))

    prefix = os.path.join(root_dir, 'buildenv')
    python_path = os.path.join(root_dir, 'deps', 'python')

    cwd = os.path.abspath(os.getcwd())
    print('Building in %s' % python_path)
    os.chdir(python_path)

    commands = [
        f'./configure --prefix={prefix}'
        # ' --enable-optimizations'
        # ' --enable-loadable-sqlite-extensions'
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

    for command in commands:
        print()
        print('\t***', command, '***')
        subprocess.run(command.split(), check=True)

    # os.chdir(root_dir)
    # install_self = 'buildenv/bin/pip3 install -e .'
    # print()
    # print('\t***', install_self, '***')
    # subprocess.run(install_self.split())

    os.chdir(cwd)


if __name__ == '__main__':
    build()
