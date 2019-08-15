# import os
import sys

from setuptools import setup, Extension
# from setuptools.command.build_ext import build_ext as build_python_ext

if not hasattr(sys, 'version_info') or sys.version_info < (3, 6):
    raise SystemExit("PyGodot requires Python version 3.6 or above.")


# class GDNativeExtension(Extension):
#     def __init__(self, name):
#         super().__init__(name, sources=[])


# if os.path.realpath(os.path.dirname(__file__)) != os.path.realpath(os.getcwd()):
#     os.chdir(os.realpath(os.path.dirname(__file__)))


# class build_ext(build_python_ext):
#     def run(self):
#         for ext in self.extensions:
#             self.build_gdnative(ext)

#     def build_gdnative(self, ext):
#         cwd = os.getcwd()
#         extension_path = self.get_ext_fullpath(ext.name)
#         if extension_path.startswith(cwd):
#             extension_path = extension_path[len(cwd):].lstrip(os.sep)

#         # Build only when the virtualenv is active
#         if not self.dry_run and 'VIRTUAL_ENV' in os.environ:
#             scons = os.path.join(sys.prefix, 'Scripts', 'scons') if sys.platform == 'win32' else 'scons'
#             args = [scons, 'target_extension=%s' % extension_path]

#             self.spawn(args)


version = __import__('godot_tools').__version__

packages = ['godot_tools']
package_data = {
    # 'godot': [
    #     '*.pxd',
    #     'cpp/*.pxd',
    #     'bindings/*.pxd',
    #     'bindings/cpp/*.pxd'
    #     'bindings/cython/*.pxd'
    # ],
    # 'godot_headers': ['*.pxd'],
    'godot_tools': [
        # TODO: Compile build templates and remove runtime dependency on Mako
        'build/templates/*.mako'
    ]
}

entry_points = {'console_scripts': ['pygodot=godot_tools.cli:pygodot', 'bindgen=godot_tools.cli:bindgen', 'pygodot_cython=godot_tools.cli:pygodot_cython']}

install_requires = [
    'Cython',
    'Mako',
    'scons',
    'pycparser',
    'autopxd2',
    'Click'
]

# setup_requires = ['scons', 'Cython', 'Mako', 'pycparser', 'autopxd2', 'Click']

setup(
    name='pygodot',
    version=version,
    python_requires='>=3.6',
    packages=packages,
    package_data=package_data,
    # cmdclass={'build_ext': build_ext},
    # ext_modules=[GDNativeExtension('_pygodot')],
    install_requires=install_requires,
    # setup_requires=setup_requires,
    entry_points=entry_points
)
