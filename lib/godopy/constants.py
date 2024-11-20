from godot.types import Color

__all__ = [
    'DEFAULT_HEIGHT',
    'DEFAULT_WIDTH',

    'KEEP',
    'DISABLED',
    'CANVAS_ITEMS',
    'VIEWPORT',

    'FRACTIONAL',
    'INTEGER',

    'JOYPAD_MOTION_RIGHT',
    'JOYPAD_MOTION_LEFT',
    'JOYPAD_MOTION_UP',
    'JOYPAD_MOTION_DOWN',

    'HINT_TEXTURE_2D',
    'HINT_PACKED_SCENE',

    'META_VERTICAL_GUIDES',
    'META_HORIZONTAL_GUIDES',
    'META_GROUP',
]

DEFAULT_HEIGHT = 648
DEFAULT_WIDTH = 1152

KEEP = 'keep'
DISABLED = 'disabled'
CANVAS_ITEMS = 'canvas_items'
VIEWPORT = 'viewport'

FRACTIONAL = 'fractional'
INTEGER = 'integer'

JOYPAD_MOTION_RIGHT = (2, 1.0)
JOYPAD_MOTION_LEFT = (2, -1.0)
JOYPAD_MOTION_UP = (1, -1.0)
JOYPAD_MOTION_DOWN = (1, 1.0)

HINT_TEXTURE_2D = 'Texture2D'
HINT_PACKED_SCENE = 'PackedScene'

META_VERTICAL_GUIDES = '_edit_vertical_guides_'
META_HORIZONTAL_GUIDES = '_edit_horizontal_guides_'
META_GROUP = '_edit_group_'
