import sys
import os

from setuptools import Extension
from .extensions import ExtType, gdnative_build_ext


class GodotProject(Extension):
    def __init__(self, name, source=None, *, binary_path='.bin'):
        if source is None:
            source = '_' + name

        if binary_path[0] != '.':
            sys.stderr.write("Binary paths should start with a \".\".\n")
            sys.exit(1)

        self.shadow_name = source
        self.binary_path = binary_path
        self._gdnative_type = ExtType.PROJECT
        super().__init__(name, sources=[])

    def get_setuptools_name(self, name, validate=None):
        dirname, fullbasename = os.path.split(name)
        basename, extension = os.path.splitext(fullbasename)

        if validate is not None and extension != validate:
            sys.stderr.write("\"%s\" extension was expected for \"%s\".\n" % (validate, name))
            sys.exit(1)

        return os.path.join(self.name, dirname, basename)


def get_cmdclass():
    return {'build_ext': gdnative_build_ext}
