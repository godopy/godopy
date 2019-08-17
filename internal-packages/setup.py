from setuptools import setup

packages = ['godot', 'godot_headers', 'godot.core', 'godot.bindings', 'godot.bindings.python']
package_data = {
    'godot': [
        '*.pxd',
        'core/*.pxd',
        'bindings/*.pxd',
        'bindings/cpp/*.pxd',
        'bindings/cython/*.pxd',
    ],
    'godot_headers': ['*.pxd'],
}

install_requires = [
    'Cython',
    'numpy',
]

setup(
    name='pygodot-internal-packages',
    version='0.0.1a',
    python_requires='>=3.8',
    packages=packages,
    package_data=package_data,
    install_requires=install_requires,
)
