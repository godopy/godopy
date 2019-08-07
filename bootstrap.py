import os
import sys
import glob
import shutil
import subprocess


from binding_generator import generate_bindings

root_dir = os.path.abspath(os.path.dirname(__file__))
headers_dir = os.path.join(root_dir, 'godot_headers')
cwd = os.path.abspath(os.getcwd())


def copy_headers():
    godot_build_dir = os.environ.get('GODOT_BUILD')
    if not godot_build_dir:
        raise SystemExit("'GODOT_BUILD' environment variable is required.")

    source_dir = os.path.join(godot_build_dir, 'modules', 'gdnative', 'include')
    print('\n*** Copying godot_headers from %r ***' % source_dir)
    shutil.copytree(source_dir, headers_dir)

    exe_glob = 'godot.*.64.exe' if sys.platform == 'win32' else 'godot.*.64'
    godot_exe_list = glob.glob(os.path.join(godot_build_dir, 'bin', exe_glob))
    if not godot_exe_list:
        raise SystemExit("Can't find Godot executable.")

    godot_exe = godot_exe_list.pop()
    api_path = os.path.join(headers_dir, 'api.json')

    print('\n*** Generating GDNative API JSON ***')
    subprocess.run([godot_exe, '--gdnative-generate-json-api', api_path], check=True)

    with open(os.path.join(headers_dir, '__init__.py'), 'w', encoding='utf-8'):
        pass  # Empty file


def build_python():
    prefix = os.path.join(root_dir, 'buildenv')
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

    if os.path.exists(headers_dir):
        print('\n*** Removing old %r ***' % headers_dir)
        shutil.rmtree(headers_dir)

    copy_headers()

    os.chdir(root_dir)
    generate_bindings(os.path.join(headers_dir, 'api.json'))
