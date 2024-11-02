import os
import traceback
import importlib

import godot
from godot.singletons import ProjectSettings
from godot.classdb import ScriptExtension


class Python(godot.Class, inherits=ScriptExtension):
    def __init__(self):
        from .register_types import python_language

        self.source_lines = []
        self.module = None
        self.error = None
        self.path = None
        self.import_path = None
        self.name = ''
        self.main_class = None

        self.extends = ''
        self.base = None

        self.language = python_language

    def _has_source_code(self):
        return len(self.source_lines) > 0

    def _get_source_code(self):
        return '\n'.join(self.source_lines)

    def _set_source_code(self, code):
        self.source_lines = code.splitlines()

    def load_source(self, f):
        self.source_lines = []
        for line in f:
            self.source_lines.append(line.rstrip())

    def save_source(self, f):
        for line in self.source_lines:
            f.write("%s\n" % line)

    def import_module(self, path):
        self.path = path
        _, inner_path = path.split('://')
        components = inner_path.split('/')
        filename = components.pop()
        name, ext = os.path.splitext(filename)

        # self.name = name
        self.import_path = '.'.join(components + [name])

        print("Importing %r" % self.import_path)
        self.module = importlib.import_module(self.import_path)

        for obj in dir(self.module):
            if isinstance(obj, type):
                self.main_class = obj
                self.name = obj.__name__
                print("Detected Script class: %r(%r) in %r" % (self.name, obj, self))
                break
        if not self.name:
            print("No script class detected in %r" % self)


    def set_error(self, exc):
        self.error = exc


    def load(self, path=None) -> godot.Error:
        if path is not None:
            self.path = path
        self.error = None

        abspath = ProjectSettings.globalize_path(path)
        print("abspath: %r" % abspath)

        try:
            with open(abspath, 'r', encoding='utf-8') as f:
                self.load_source(f)
        except Exception as exc:
            traceback.print_exception(exc)
            godot.push_error(str(exc))
            self.set_error(exc)
            return godot.Error.FAIL

        try:
            self.import_module(path)
        except Exception as exc:
            traceback.print_exception(exc)
            godot.push_error(str(exc))
            self.set_error(exc)
            return godot.Error.FAIL

        print('Resource %r loaded from %r and imported to %r' % (self, self.path, self.module))

        self.language.add_script(self)

        return godot.Error.OK


    def save(self, path=None) -> godot.Error:
        if path is not None:
            self.path = path

        self.error = None
        print("Saving %r to %r" % (self, path))

        abspath = ProjectSettings.globalize_path(path)
        print("abspath: %r" % abspath)

        try:
            with open(abspath, 'w', encoding='utf-8') as f:
                self.save_source(f)
        except Exception as exc:
            traceback.print_exception(exc)
            godot.push_error(str(exc))
            self.set_error(exc)
            return godot.Error.FAIL

        print('Resource %r (%r) saved to %r' % (self, self.module, self.path))

        return godot.Error.OK

    def _update_exports(self):
        pass

    def _get_documentation(self):
        pass

    def _reload(self, keep_state):
        if not self.path:
            godot.push_error("No script path defined")
            return

        self.load(self.path)

    def _get_language(self):
        return self.language

    def _is_valid(self):
        print("_is_valid call %r %r" % (self.error is None, self.module is not None))
        return self.error is None and self.module is not None

    def _can_instantiate(self):
        return self._is_valid()

    def _is_tool(self):
        return False

    def _get_instance_base_type(self):
        if self.extends:
            return self.extends

        return ''

    def _get_base_script(self):
        return self.base

    def _get_global_name(self):
        return self.name

    def _get_script_method_list(self):
        return []

    def _has_method(self, name):
        return False

    def _has_static_method(self, name):
        return False

    def _get_method_info(self, name):
        return {}

    def _get_script_property_list(self):
        return []

    def _get_members(self):
        return []

    def _has_property_default_value(self, prop_name):
        return False

    def _has_script_signal(self, signal_name):
        return False

    def _get_script_signal_list(self):
        return []

    def _get_rpc_config(self):
        return {}

    def _get_constants(self):
        return {}

    def _instance_create(self, for_object):
        pass
        # TODO: gdextension_interface_script_instance_create3

    def _instance_has(self, obj):
        return False
