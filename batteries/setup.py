from setuptools import setup

packages = [
    'godot_headers',
    'godot', 'godot.core', 'godot.bindings', 'godot.bindings.python',
    'godopy',
    'build_tools'
]

package_data = {
    'godot_headers': ['*.pxd'],
    'godot': ['*.pxd'],
    'godopy': ['*.pxd'],
    'godot.core': ['*.pxd'],
    'godot.bindings': [
        '*.pxd',
        'cpp/*.pxd',
        'cython/*.pxd',
    ]
}

install_requires = [
    'Cython',
    'numpy',
]

entry_points = {'console_scripts': ['godopy_cython=build_tools:godopy_cython']}


setup(
    name='godopy-batteries',
    version='0.0.1a',
    python_requires='>=3.8',
    packages=packages,
    entry_points=entry_points,
    package_data=package_data,
    install_requires=install_requires,
)
