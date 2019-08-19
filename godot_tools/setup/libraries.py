from setuptools import Extension
from .enums import ExtType


class GenericGDNativeLibrary(Extension):
    def __init__(self, name, **gdnative_options):
        self._gdnative_type = ExtType.GENERIC_LIBRARY
        self._gdnative_options = gdnative_options

        super().__init__(name, sources=[])


class GDNativeLibrary(Extension):
    def __init__(self, name, *, extra_sources=None, **gdnative_options):
        self._gdnative_type = ExtType.LIBRARY
        self._gdnative_options = gdnative_options

        sources = []

        if extra_sources is not None:
            for src in extra_sources:
                sources.append(src)

        super().__init__(name, sources=sources)
