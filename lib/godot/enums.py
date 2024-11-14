import enum

import gdextension

__all__ = [
    'ClockDirection',
    'Corner',
    'Error',
    'EulerOrder',
    'HorizontalAlignment',
    'InlineAlignment',
    'JoyAxis',
    'JoyButton',
    'Key',
    'KeyLocation',
    'KeyModifierMask',
    'MIDIMessage',
    'MethodFlags',
    'MouseButton',
    'MouseButtonMask',
    'Orientation',
    'PropertyHint',
    'PropertyUsageFlags',
    'Side',
    'VariantOperator',
    'VariantType',
    'VerticalAlignment'
]


class _classdict(dict):
    pass


def _set_globals():
    for enum_name, enum_data_list in gdextension.global_enums_dict().items():
        enum_name = enum_name.replace('.', '')
        enum_data = _classdict(enum_data_list)
        enum_data._member_names = enum_data.keys()
        globals()[enum_name] = enum.EnumType(enum_name, (enum.IntEnum,), enum_data)


_set_globals()
