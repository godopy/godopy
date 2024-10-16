import enum

import gdextension as gde
from . import classdb, singletons
from .core import *


input = gde.input


__all__ = [
    'classdb',
    'singletons',

    'input',

    'MODULE_INITIALIZATION_LEVEL_CORE',
    'MODULE_INITIALIZATION_LEVEL_EDITOR',
    'MODULE_INITIALIZATION_LEVEL_SCENE',
    'MODULE_INITIALIZATION_LEVEL_SERVERS',

    'method',
    'virtual_method',

    'GodotClassBase',
    'Class',
    'EngineClass',
    'EngineObject',
    'Extension',

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
    'VerticalAlignment',

    'abs',
    'absf',
    'absi',
    'acos',
    'acosh',
    'angle_difference',
    'asin',
    'asinh',
    'atan',
    'atan2',
    'atanh',
    'bezier_derivative',
    'bezier_interpolate',
    'ceil',
    'ceilf',
    'ceili',
    'clamp',
    'clampf',
    'clampi',
    'cos',
    'cosh',
    'cubic_interpolate',
    'cubic_interpolate_angle',
    'cubic_interpolate_angle_in_time',
    'cubic_interpolate_in_time',
    'db_to_linear',
    'deg_to_rad',
    'ease',
    'error_string',
    'exp',
    'floor',
    'floorf',
    'floori',
    'fmod',
    'fposmod',
    'hash',
    'instance_from_id',
    'inverse_lerp',
    'is_equal_approx',
    'is_finite',
    'is_inf',
    'is_instance_id_valid',
    'is_instance_valid',
    'is_nan',
    'is_same',
    'is_zero_approx',
    'lerp',
    'lerp_angle',
    'lerpf',
    'linear_to_db',
    'log',
    'max',
    'maxf',
    'maxi',
    'min',
    'minf',
    'mini',
    'move_toward',
    'nearest_po2',
    'pingpong',
    'posmod',
    'pow',
    'print',
    'print_rich',
    'print_verbose',
    'printerr',
    'printraw',
    'prints',
    'printt',
    'push_error',
    'push_warning',
    'rad_to_deg',
    'rand_from_seed',
    'randf',
    'randf_range',
    'randfn',
    'randi',
    'randi_range',
    'randomize',
    'remap',
    'rid_allocate_id',
    'rid_from_int64',
    'rotate_toward',
    'round',
    'roundf',
    'roundi',
    'seed',
    'sign',
    'signf',
    'signi',
    'sin',
    'sinh',
    'smoothstep',
    'snapped',
    'snappedf',
    'snappedi',
    'sqrt',
    'step_decimals',
    'str',
    'tan',
    'tanh',
    'type_convert',
    'type_string',
    'typeof',
    'weakref',
    'wrap',
    'wrapf',
    'wrapi',
]


_SKIP_UTILITY_FUNCTION = [
    'bytes_to_var',
    'bytes_to_var_with_objects',
    'str_to_var',
    'var_to_bytes',
    'var_to_bytes_with_objects',
    'var_to_str',
]


class _classdict(dict):
    pass


def _set_globals():
    for func_name in gde._utility_functions_dir():
        if func_name in _SKIP_UTILITY_FUNCTION:
            continue
        globals()[func_name] = gde.UtilityFunction(func_name)

    for enum_name, enum_data_list in gde._enums_dir().items():
        enum_name = enum_name.replace('.', '')
        enum_data = _classdict(enum_data_list)
        enum_data._member_names = enum_data.keys()
        globals()[enum_name] = enum.EnumType(enum_name, (enum.IntEnum,), enum_data)


MODULE_INITIALIZATION_LEVEL_CORE = 0
MODULE_INITIALIZATION_LEVEL_SERVERS = 1
MODULE_INITIALIZATION_LEVEL_SCENE = 2
MODULE_INITIALIZATION_LEVEL_EDITOR = 3


_set_globals()
