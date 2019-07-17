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

class build_ext(build_python_ext):
    def run(self):
        for ext in self.extensions:
            self.build_gdnative(ext)

    def build_gdnative(self, ext):
        cwd = os.getcwd()
        extension_path = self.get_ext_fullpath(ext.name)
        if extension_path.startswith(cwd):
            extension_path = extension_path[len(cwd):].lstrip(os.sep)
        print(extension_path)
        if not self.dry_run:
            args = ['scons', f'target_extension={extension_path}']
            if generate_bindings:
                args += ['generate_bindings=yes']
            self.spawn(args)


version = __import__('godot').__version__

packages = ['godot']
package_data = {
    'godot': [
        '/*.pxd',
        'headers/*.pxd',
        'cli/templates/*.mako',
        'cpp_interop/templates/*.mako'
    ]
}

entry_points = {'console_scripts': 'pygodot=godot.cli:pygodot'}

install_requires = [
    'redbaron',
    'autopxd2',
    'pycparser',
    'Click',
    'Cython',
    'Mako'
]

setup_requires = ['scons']

setup_args = dict(
    name='pygodot',
    version=version,
    python_requires='>=3.7',
    packages=packages,
    package_data=package_data,
    ext_modules=[GDNativeExtension('pygodot')],
    cmdclass={'build_ext': build_ext},
    install_requires=install_requires,
    setup_requires=setup_requires,
    entry_points=entry_points
)

setup(**setup_args)
