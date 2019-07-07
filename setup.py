from setuptools import find_packages, setup

version = __import__('godot').get_version()

setup(
    name='godot-cython',
    version=version,
    python_requires='>=3.7',
    packages=find_packages(),
    package_data={
        'godot-cython': ['godot/*.pxd']
    }
)
