import sys
from setuptools import setup

version = __import__('godot').get_version()

packages = ['godot', 'godot_cpp', 'godot_headers']
package_data = {
    'godot': ['data/runtime-support/*.dylib', 'data/runtime-support/*.a', 'data/runtime-support/*.py'],
    'godot_cpp': ['*.pxd'],
    'godot_headers': ['*.pxd']
}

entry_points = '''
[console_scripts]
pygodot=godot.cli:pygodot
'''

install_requires = [
    'redbaron',
    'autopxd2',
    'pycparser',
    'Click',
    'Cython',
    'Mako'
]

# Dev symlinks:
# cd <godot-project-dir>/bin/<platform>
# ln -s ../../../pygodot/dist/pygodot.app/Contents/Resources pyres

setup_requires = []

loader_module = ['src/pylib/__loader__.py']

setup_args = dict(
    name='pygodot',
    version=version,
    python_requires='>=3.7',
    packages=packages,
    package_data=package_data,
    install_requires=install_requires,
    setup_requires=setup_requires,
    entry_points=entry_points
)

if sys.platform == 'darwin':
    setup_requires.append('py2app')
    setup_args['app'] = loader_module

setup(**setup_args)
