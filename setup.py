from setuptools import setup

version = __import__('godot').get_version()

setup(
    name='pygodot',
    version=version,
    python_requires='>=3.7',
    packages=['godot', 'godot_cpp', 'godot_headers'],
    package_data={
        'godot_cpp': ['*.pxd'],
        'godot_headers': ['*.pxd']
    },
    install_requires=[
        'redbaron',
        'autopxd2',
        'pycparser',
        'Click',
        'Cython',
        'Mako'
    ],
    entry_points='''
    [console_scripts]
    pygodot=godot.cli:pygodot
    '''
)
