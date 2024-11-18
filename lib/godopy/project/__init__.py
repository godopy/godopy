from typing import Any, List, Mapping, Optional, Tuple

import godot
from godot.classdb import (
    ProjectSettings,
    InputEventKey, InputEventJoypadMotion, InputEventJoypadButton
)


def make_input_setting(*, keys: Optional[List[int]] = None,
                          joypad_buttons: Optional[List[int]] = None,
                          joypad_motions: Optional[List[Tuple[int, float]]] = None,
                          deadzone: float = 0.2):
    events = []

    if keys is not None:
        for key in keys:
            input_event_key = InputEventKey()
            input_event_key.set_physical_keycode(key)
            events.append(input_event_key)
    if joypad_buttons is not None:
        for button in joypad_buttons:
            input_event_joypad_button = InputEventJoypadButton()
            input_event_joypad_button.set_button_index(button)
            events.append(input_event_joypad_button)
    if joypad_motions is not None:
        for axis, value in joypad_motions:
            input_event_joypad_motion = InputEventJoypadMotion()
            input_event_joypad_motion.set_axis(axis)
            input_event_joypad_motion.set_axis_value(value)
    return {
        'deadzone': deadzone,
        'events': events
    }


_aliases = {
    'name': 'application/config/name',
    'application/name': 'application/config/name',
    'scene': 'application/run/main_scene',
    'main_scene': 'application/run/main_scene',
    'application/main_scene': 'application/run/main_scene',
    'icon': 'application/config/icon',
    'application/icon': 'application/config/icon',

    'width': 'display/window/size/viewport_width',
    'height': 'display/window/size/viewport_height',
    'display/width': 'display/window/size/viewport_width',
    'display/height': 'display/window/size/viewport_width',
    'width_override': 'display/window/size/window_width_override',
    'height_override': 'display/window/size/window_height_override',
    'display/width_override': 'display/window/size/window_width_override',
    'display/height_override': 'display/window/size/window_height_override',
    'type': 'rendering/renderer/rendering_method',
    'rendering/rendering_method': 'rendering/renderer/rendering_method',
    'stretch_mode': 'display/window/stretch/mode',
    'display/stretch_mode': 'display/window/stretch/mode',

    'snap_2d_transforms_to_pixel': 'rendering/2d/snap/snap_2d_transforms_to_pixel',
    'snap_2d_vertices_to_pixel': 'rendering/2d/snap/snap_2d_vertices_to_pixel',
    'rendering/snap_2d_transforms_to_pixel': 'rendering/2d/snap/snap_2d_transforms_to_pixel',
    'rendering/snap_2d_vertices_to_pixel': 'rendering/2d/snap/snap_2d_vertices_to_pixel',
}


class Project:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        if config is None:
            config = {}

        self.name = Project.get_setting('application/config/name') or '[Untitled]'

        for key, value in config.items():
            setting = _aliases.get(key, key)

            if not ProjectSettings.has_setting(setting):
                raise AttributeError(f"Setting {setting!r} does not exist")

            existing = Project.get_setting(setting)

            if existing != value:
                ProjectSettings.set_setting(setting, value)

            if setting == 'application/config/name':
                self.name = value

        ProjectSettings.save()

    def __repr__(self):
        return f"<GodoPy project {self.name!r}>"


    @staticmethod
    def get_setting(setting: str, default: Optional[str] = None, *, with_override: bool = False):
        setting = _aliases.get(setting, setting)

        if not ProjectSettings.has_setting(setting):
            raise AttributeError(f"Setting {setting!r} does not exist")

        if with_override:
            return ProjectSettings.get_setting_with_override(setting, default)
        return ProjectSettings.get_setting(setting, default)


    @staticmethod
    def save(custom_file=None) -> None:
        if custom_file is not None:
            result = ProjectSettings.save_custom(custom_file)
        else:
            result = ProjectSettings.save()

        if result != godot.Error.OK:
            err = godot.Error.from_bytes([result])
            raise RuntimeError(f"Could not save Project settings, the error was: {err}")
