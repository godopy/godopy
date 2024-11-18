from typing import Any, List, Mapping, Optional, Tuple

import godot
from godot.classdb import (
    ProjectSettings,
    InputEventKey, InputEventJoypadMotion, InputEventJoypadButton
)


def make_input_setting(*, keys: Optional[List[int | Tuple[int, str]]] = None,
                          joypad_buttons: Optional[List[int]] = None,
                          joypad_motions: Optional[List[Tuple[int, float]]] = None,
                          deadzone: float = 0.2):
    events = []

    if keys is not None:
        for key in keys:
            input_event_key = InputEventKey()
            if isinstance(key, int):
                input_event_key.set_physical_keycode(key)
            elif hasattr(key, '__len__') and len(key) == 2:
                _key, s = key
                if len(s) != 1:
                    raise ValueError('Unicode value for InputEventKey must be a single character')
                input_event_key.set_physical_keycode(_key)
                input_event_key.set_unicode(ord(s))
            else:
                raise ValueError(f"Incorrect 'InputEventKey' setting: {key!r}")
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
    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        input_map: Optional[Mapping[str, Any]] = None
    ) -> None:
        if config is None:
            config = {}
        if input_map is None:
            input_map = {}

        self.name = Project.get_setting('application/config/name') or '[Untitled]'

        # TODO: Sync settings only inside Editor or with some command line argument

        updated = False

        for key, value in config.items():
            setting = _aliases.get(key, key)

            if not ProjectSettings.has_setting(setting):
                raise AttributeError(f"Setting {setting!r} does not exist")

            existing = Project.get_setting(setting)

            if existing != value:
                ProjectSettings.set_setting(setting, value)
                updated = True

            if setting == 'application/config/name':
                self.name = value

        for key, value in input_map.items():
            setting = f'input/{key}'

            if not ProjectSettings.has_setting(setting):
                raise AttributeError(f"Setting {setting!r} does not exist")

            existing = Project.get_setting(setting)

            if existing != value:
                ProjectSettings.set_setting(setting, value)
                updated = True

        # TODO: Store created settings somewhere to be able to reset them automatically
        #       when they are gone from Python settings

        if updated:
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
            raise RuntimeError(f"Could not save Project settings, the error was: {godot.Error(result)}")
