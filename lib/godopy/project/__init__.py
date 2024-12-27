import sys
from typing import Any, List, Mapping, Optional, Tuple

import godot
from godot.classdb import (
    Engine, ProjectSettings,
    InputEventKey, InputEventJoypadMotion, InputEventJoypadButton
)

from ..scene import Scene


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
    'rendering/snap_2d_transforms_to_pixel': 'rendering/2d/snap/snap_2d_transforms_to_pixel',

    'snap_2d_vertices_to_pixel': 'rendering/2d/snap/snap_2d_vertices_to_pixel',
    'rendering/snap_2d_vertices_to_pixel': 'rendering/2d/snap/snap_2d_vertices_to_pixel',
}


UNTITLED = '[Untitled]'

class Project:
    def __init__(
        self,
        name: str = UNTITLED,
        main_scene: Optional[Scene] = None,
        config: Optional[Mapping[str, Any]] = None,
        input_map: Optional[Mapping[str, Any]] = None
    ) -> None:
        if config is None:
            config = {}
        if input_map is None:
            input_map = {}

        existing_name = Project.get_setting('application/config/name')
        self.name = name
        if self.name == UNTITLED and existing_name:
            self.name = existing_name

        existing_main_scene_path = Project.get_setting('application/run/main_scene')

        self.main_scene = main_scene
        if self.main_scene is None and existing_main_scene_path:
            self.main_scene = Scene.from_path(existing_main_scene_path)

        updated = False

        if Engine.is_editor_hint():
            # Sync settings only when the Editor is active
            if name != UNTITLED and existing_name != name:
                Project.set_setting('name', name)

            if main_scene is not None:
                main_scene_path = main_scene.get_path()
                if main_scene_path != existing_main_scene_path:
                    Project.set_setting('main_scene', main_scene_path)

            for setting, value in config.items():
                setting = _aliases.get(setting, setting)

                if isinstance(value, Scene):
                    value = value.get_path()

                existing = Project.get_setting(setting)

                if existing != value:
                    Project.set_setting(setting, value)
                    updated = True

                if setting == 'application/config/name':
                    self.name = value

            for key, value in input_map.items():
                setting = f'input/{key}'

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
    def get_setting(setting: str, default: Optional[str] = None, *,
                    with_override: bool = False, instantiate_scenes=False) -> Any:
        setting = _aliases.get(setting, setting)

        if not ProjectSettings.has_setting(setting):
            return default

        if with_override:
            value = ProjectSettings.get_setting_with_override(setting, default)
        else:
            value = ProjectSettings.get_setting(setting, default)

        if instantiate_scenes and setting.endswith('scene') and value.startswith('res://'):
            value = Scene.from_path(value)

        return value

    @staticmethod
    def set_setting(setting: str, value: Any) -> None:
        setting = _aliases.get(setting, setting)

        ProjectSettings.set_setting(setting, value)

    @staticmethod
    def has_setting(setting: str, value: Any) -> bool:
        setting = _aliases.get(setting, setting)

        return ProjectSettings.has_setting(setting)

    @staticmethod
    def save(custom_file=None) -> None:
        if custom_file is not None:
            result = ProjectSettings.save_custom(custom_file)
        else:
            result = ProjectSettings.save()

        if result != godot.Error.OK:
            raise RuntimeError(f"Could not save Project settings, the error was: {godot.Error(result)}")
