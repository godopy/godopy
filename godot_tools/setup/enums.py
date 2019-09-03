import enum


class ExtType(enum.Enum):
    PROJECT = enum.auto()
    ADDON = enum.auto()
    GENERIC_LIBRARY = enum.auto()
    LIBRARY = enum.auto()
    NATIVESCRIPT = enum.auto()
