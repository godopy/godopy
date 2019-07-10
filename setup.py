from setuptools import setup

version = __import__('pygdnlib').get_version()

setup(
    name='godot-cython',
    version=version,
    python_requires='>=3.7',
    packages=['pygdnlib', 'godot_cpp', 'godot_headers'],
    package_data={
        'godot_cpp': ['*.pxd'],
        'godot_headers': ['*.pxd']
    }
)
