import os
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_python_ext

if not hasattr(sys, 'version_info') or sys.version_info < (3, 7):
    raise SystemExit("PyGodot requires Python version 3.7 or above.")


class GDNativeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


generate_bindings = False
if '--generate_bindings' in sys.argv:
    sys.argv.remove('--generate_bindings')
    generate_bindings = True

export_build = False
if '--export' in sys.argv:
    sys.argv.remove('--export')
    export_build = True


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
            args = ['scons', 'target_extension=%s' % extension_path]
            if generate_bindings:
                args += ['generate_bindings=yes']
            if export_build:
                args += ['export=yes']
            self.spawn(args)


version = __import__('pygodot').__version__

packages = ['pygodot', 'godot_headers']
package_data = {
    'godot': [
        '/*.pxd',
        'bindings/*.pxd',
        'bindings/core/*.pxd',
        'bindings/cpp/*.pxd',

        'templates/*.mako',
        # TODO: Compile build templates and remove runtime dependency on Mako
        'build/templates/*.mako'
    ],
    'godot_headers': ['/*.pxd']
}

entry_points = {'console_scripts': 'pygodot=pygodot.cli:pygodot'}

install_requires = [
    'autopxd2',
    'pycparser',
    'Click',
    'Cython',
    'Mako'
]

setup_requires = ['scons', 'Cython']

setup_args = dict(
    name='pygodot',
    version=version,
    python_requires='>=3.7',
    packages=packages,
    package_data=package_data,
    cmdclass={'build_ext': build_ext},
    install_requires=install_requires,
    setup_requires=setup_requires,
    entry_points=entry_points
)


headers_def = os.path.join(os.getcwd(), 'godot_headers', 'gdnative_api.pxd')
print(headers_def)
if os.path.exists(headers_def):
    setup_args['ext_modules'] = [GDNativeExtension('_pygodot')]

setup(**setup_args)
