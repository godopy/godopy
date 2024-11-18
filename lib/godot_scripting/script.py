import os
import importlib
from typing import Any, Dict, List


import godot
import gdextension
from godot import types
from godot.classdb import Engine, ProjectSettings, Resource, ScriptExtension


class PythonProjectSettings(godot.Class, inherits=Resource):
    pass


class PythonExtension(godot.Class, inherits=Resource):
    pass

class PythonScriptInstance(gdextension.ScriptInstance):
    def __init__(self, script: ScriptExtension, for_object: gdextension.Object) -> None:
        super().__init__(script, for_object, script._script_dict)

    def is_placeholder(self) -> bool:
        print("is_placeholder")
        return False

    def notification(self, what, reversed=False):
        pass


class PythonPlaceholderScriptInstance(gdextension.ScriptInstance):
    def __init__(self, script: ScriptExtension, for_object: gdextension.Object) -> None:
        super().__init__(script, for_object, {})

    def is_placeholder(self):
        print("placeholder is_placeholder")
        return True


class PythonScript(godot.Class, inherits=ScriptExtension, no_virtual_underscore=True):
    def __init__(self):
        from .register_types import python_language

        self._source = ''
        self._module = None

        self._error = None
        self._resource_path = None
        self._import_path = None

        self._script_dict = {}

        self.name = ''

        self.extends = None
        self.base_script = None
        self.instance = None

        self.language = python_language

        self._method_info_list = []
        self._method_info_dict = {}

    def editor_can_reload_from_file(self):
        print("editor_can_reload_from_file")
        return True

    def placeholder_erased(self, placeholder):
        print("placeholder_erased", placeholder)

    def can_instantiate(self):
        return not Engine.is_editor_hint()

    def get_base_script(self):
        print("get_base_script")
        return self.base_script

    def get_global_name(self):
        return self.name

    def inherits_script(self, script) -> bool:
        print("inherits_script", script)
        return False

    def get_instance_base_type(self):
        if self.extends is not None:
            return self.extends.__name__

        return ''

    def instance_create(self, for_object: godot.Class) -> PythonScriptInstance:
        self.instance = PythonScriptInstance(self, for_object)

        self._method_info_list = [m.as_dict() for m in self.instance.get_method_list()]
        self._method_info_dict = {}
        for m in self._method_info_list:
            self._method_info_dict[m['name']] = m

        return self.instance

    def placeholder_instance_create(self, for_object: godot.Class) -> PythonPlaceholderScriptInstance:
        self.instance = PythonPlaceholderScriptInstance(self, for_object)

        return self.instance

    def instance_has(self, object):
        print("instance_has", object)
        return False

    def has_source_code(self) -> bool:
        return bool(self._source)

    def get_source_code(self):
        return self._source

    def set_source_code(self, code: str) -> None:
        self._source = code

    def _load_source(self, f):
        self._source = f.read()

    def _save_source(self, f):
        f.write(self._source)

    def _import_module(self, path):
        self._resource_path = path

        _, inner_path = path.split('://')
        components = inner_path.split('/')
        filename = components.pop()
        name, _ = os.path.splitext(filename)

        self._import_path = '.'.join(components + [name])
        self._module = importlib.import_module(self._import_path)

        self._script_dict.update(self._module.__dict__)

        for key in list(self._script_dict.keys()):
            if key == '__class_name__':
                self.name = self._script_dict.pop(key)
            elif key == '__extends__':
                self.extends = self._script_dict.pop(key)
            elif key.startswith('__'):
                del self._script_dict[key]

        if not self.name:
            self.name = name.title().replace('_', '')

        if self.extends is None:
            # If a script does not define '__extends__' it is interprered as a SceneTree
            # with one '_initialize' method where the whole script is executed with
            # a global variable '__name__' being equal to '__main__'.
            #
            # This feature emulates the behavior of normal Python scripts.
            from godot.classdb import SceneTree
            self.extends = SceneTree

            source = self.get_source_code()
            def _initialize(self) -> None:
                try:
                    exec(source, {'__name__': '__main__'})
                except Exception as exc:
                    gdextension.print_script_error_with_traceback(exc)
                    self.quit(1)

                self.quit(0)

            self._script_dict = {
                '_initialize': _initialize
            }


    def _set_error(self, exc):
        self._error = exc


    def _load(self, path=None) -> godot.Error:
        if path is not None:
            self._resource_path = path
        self._error = None

        abspath = ProjectSettings.globalize_path(path)

        try:
            with open(abspath, 'r', encoding='utf-8') as f:
                self._load_source(f)
        except Exception as exc:
            gdextension.print_script_error_with_traceback(exc)
            self._set_error(exc)
            return godot.Error.FAILED

        try:
            self._import_module(path)
        except Exception as exc:
            gdextension.print_script_error_with_traceback(exc)
            self._set_error(exc)
            return godot.Error.FAILED

        print('Resource %r loaded from %r and imported to %r' % (self, self._resource_path, self._module))

        self.language._add_script(self)

        return godot.Error.OK


    def _save(self, path=None) -> godot.Error:
        if path is not None:
            self._resource_path = path

        self._error = None
        abspath = ProjectSettings.globalize_path(path)

        try:
            with open(abspath, 'w', encoding='utf-8') as f:
                self._save_source(f)
        except Exception as exc:
            gdextension.print_script_error_with_traceback(exc)
            self._set_error(exc)
            return godot.Error.FAILED

        print('Resource %r (%r) saved to %r' % (self, self._module, self._resource_path))

        return godot.Error.OK

    def reload(self, keep_state):
        if not self._resource_path:
            gdextension.print_error("No script path defined")
            return

        self._load(self._resource_path)

    def get_documentation(self):
        print("get_documentation")
        return {}

    def get_class_icon_path(self) -> str:
        print("get_class_icon_path")
        return ''

    def has_method(self, name):
        print("has_method", name)
        return name in self._method_info_dict

    def has_static_method(self, name):
        print("has_static_method", name)
        return False

    def get_script_method_argument_count(self, method: str):
        print("get_script_method_argument_count", method)
        return 0

    def get_method_info(self, name):
        print("get_method_info", name)
        return self._method_info_dict.get(name, {})

    def is_tool(self):
        return False

    def is_valid(self):
        print("is_valid call %r %r" % (self._error is None, self.module is not None))
        return self._error is None and self._module is not None

    def is_abstract(self):
        # print("is_abstract")
        return False

    def get_language(self):
        return self.language

    def has_script_signal(self, sig: str):
        print("get_script_signal", sig)
        return False

    def get_script_signal_list(self) -> List[Dict]:
        return []

    def has_property_default_value(self, prop_name) -> bool:
        print("has_prop_default_value", prop_name)
        return False

    def get_property_default_value(self, prop_name) -> Any:
        print("get_prop_default_value", prop_name)
        return None

    def update_exports(self):
        print("update_exports")

    def get_script_method_list(self):
        print("get_script_method_list")
        return self._method_info_list

    def get_script_property_list(self):
        print("get_script_property_list")
        return []

    def get_member_line(self, member: str) -> int:
        return -1

    def get_constants(self) -> Dict:
        return {}

    def get_members(self) -> List[types.StringName]:
        print("get_members")
        return []
