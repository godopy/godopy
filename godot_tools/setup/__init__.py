import sys
import os

from setuptools import setup, Extension
from .enums import ExtType
from .extensions import GDNativeBuildExt


def godot_setup(godot_project, *, library, extensions, addons=None, **kwargs):
    if addons is None:
        addons = ()
    modules = [
        GodotProject(godot_project, **kwargs),
        library,
        *addons,
        *extensions
    ]

    if len(sys.argv) < 2 or sys.argv[1] not in ('install', 'develop'):
        raise SystemExit('Usage: python godot_setup.py install [options]')

    sys.argv = [sys.argv[0], 'build_ext', '-i'] + sys.argv[2:]
    setup(ext_modules=modules, cmdclass={'build_ext': GDNativeBuildExt})


class DummyModule:
    def __init__(self):
        pass


class GodotProject(Extension):
    def __init__(self, name, python_package=None, *, binary_path=None, set_development_path=False, **kwargs):
        dir, name = os.path.split(name)

        project_module = kwargs.pop('project_module', DummyModule())

        if python_package is None:
            python_package = name if dir else '_' + name

        if binary_path is None:
            binary_path = '.bin'

        self.python_package = python_package
        self.binary_path = binary_path
        self.development_path = kwargs.get('development_path', None)
        self.set_development_path = set_development_path or self.development_path
        self._gdnative_type = ExtType.PROJECT

        self.module = project_module

        self.path_prefix = os.path.join(dir, name)
        self.project_name = kwargs.get('project_name', name)

        super().__init__(os.path.join(dir, name, 'project.godot'), sources=[])

    def get_setuptools_name(self, name, addon_prefix=None, validate=None):
        dirname, fullbasename = os.path.split(name)
        basename, extension = os.path.splitext(fullbasename)

        if validate is not None and extension != validate:
            sys.stderr.write("\"%s\" extension was expected for \"%s\".\n" % (validate, name))
            sys.exit(1)

        parts = [self.path_prefix]
        if addon_prefix:
            parts += ['addons', addon_prefix]

        if dirname:
            parts.append(dirname)

        return os.path.join(*parts, basename)
