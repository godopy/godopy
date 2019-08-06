import os
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_python_ext

if not hasattr(sys, 'version_info') or sys.version_info < (3, 7):
    raise SystemExit("PyGodot requires Python version 3.7 or above.")


class GDNativeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


if os.path.realpath(os.path.dirname(__file__)) != os.path.realpath(os.getcwd()):
    os.chdir(os.realpath(os.path.dirname(__file__)))

generate_bindings = False
headers_dir = os.path.join(os.getcwd(), 'godot_headers')
headers_def = os.path.join(headers_dir, 'gdnative_api.pxd')

if not os.path.exists(headers_def):
    generate_bindings = True

godot_build_dir = os.environ.get('GODOT_BUILD')

if not os.path.exists(headers_dir):
    import shutil
    import subprocess
    import glob

    godot_build_dir = os.environ.get('GODOT_BUILD')
    if not godot_build_dir:
        raise SystemExit("'GODOT_BUILD' environment variable is required.")

    source_dir = os.path.join(godot_build_dir, 'modules', 'gdnative', 'include')
    shutil.copytree(source_dir, headers_dir)

    exe_glob = 'godot.*.64.exe' if sys.platform == 'win32' else 'godot.*.64'
    godot_exe_list = glob.glob(os.path.join(godot_build_dir, 'bin', exe_glob))
    if not godot_exe_list:
        raise SystemExit("Can't find Godot executable.")

    godot_exe = godot_exe_list.pop()
    api_path = os.path.join(headers_dir, 'api.json')
    print(godot_exe, api_path)
    # print([godot_exe, '--gdnative-generate-json-api', api_path])
    subprocess.run([godot_exe, '--gdnative-generate-json-api', api_path], check=True)


class build_ext(build_python_ext):
    def run(self):
        for ext in self.extensions:
            self.build_gdnative(ext)

    def build_gdnative(self, ext):
        cwd = os.getcwd()
        extension_path = self.get_ext_fullpath(ext.name)
        if extension_path.startswith(cwd):
            extension_path = extension_path[len(cwd):].lstrip(os.sep)

        # Build only when the virtualenv is active
        if not self.dry_run and 'VIRTUAL_ENV' in os.environ:
            scons = os.path.join(sys.prefix, 'Scripts', 'scons') if sys.platform == 'win32' else 'scons'
            args = [scons, 'target_extension=%s' % extension_path, 'target=release']
            if generate_bindings:
                args += ['generate_bindings=yes']

            self.spawn(args)

version = __import__('pygodot').__version__

packages = ['godot', 'godot_headers', 'pygodot']
package_data = {
    'godot': [
        '*.pxd',
        'cpp/*.pxd',
        'bindings/*.pxd',
        'bindings/cpp/*.pxd'
        'bindings/cython/*.pxd'
    ],
    'godot_headers': ['*.pxd'],
    'pygodot': [
        # TODO: Compile build templates and remove runtime dependency on Mako
        'build/templates/*.mako'
    ]
}

entry_points = {'console_scripts': ['pygodot=pygodot.cli:pygodot', 'bindgen=pygodot.cli:binding_generator']}

install_requires = [
    'Cython',
    'Click',
    'Mako'
]

setup_requires = ['scons', 'Cython', 'Mako', 'pycparser', 'autopxd2', 'Click']

setup(
    name='pygodot',
    version=version,
    python_requires='>=3.7',
    packages=packages,
    package_data=package_data,
    cmdclass={'build_ext': build_ext},
    ext_modules=[GDNativeExtension('_pygodot')],
    install_requires=install_requires,
    setup_requires=setup_requires,
    entry_points=entry_points
)
