import os
import importlib
import traceback

import godot as gd
from godot import classdb, singletons as gds


class PythonScript(gd.Class, inherits=classdb.ScriptExtension):
    def __init__(self):
        self.source_lines = []
        self.module = None
        self.error = None
        self.path = None
        self.import_path = None
        self.name = ''
        self.main_class = None

        self.extends = ''
        self.base = None

        self.language = PythonLanguage.get_singleton()

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


    def load(self, path=None) -> gd.Error:
        if path is not None:
            self.path = path
        self.error = None

        abspath = gds.ProjectSettings.globalize_path(path)
        print("abspath: %r" % abspath)

        try:
            with open(abspath, 'r', encoding='utf-8') as f:
                self.load_source(f)
        except Exception as exc:
            traceback.print_exception(exc)
            gd.push_error(str(exc))
            self.set_error(exc)
            return gd.Error.FAIL

        try:
            self.import_module(path)
        except Exception as exc:
            traceback.print_exception(exc)
            gd.push_error(str(exc))
            self.set_error(exc)
            return gd.Error.FAIL

        print('Resource %r loaded from %r and imported to %r' % (self, self.path, self.module))

        self.language.add_script(self)

        return gd.Error.OK


    def save(self, path=None) -> gd.Error:
        if path is not None:
            self.path = path

        self.error = None
        print("Saving %r to %r" % (self, path))

        abspath = gds.ProjectSettings.globalize_path(path)
        print("abspath: %r" % abspath)

        try:
            with open(abspath, 'w', encoding='utf-8') as f:
                self.save_source(f)
        except Exception as exc:
            traceback.print_exception(exc)
            gd.push_error(str(exc))
            self.set_error(exc)
            return gd.Error.FAIL

        print('Resource %r (%r) saved to %r' % (self, self.module, self.path))

        return gd.Error.OK

    def _update_exports(self):
        pass

    def _get_documentation(self):
        pass

    def _reload(self, keep_state):
        if not self.path:
            gd.push_error("No script path defined")
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
        if self.extends and gds.ClassDB.class_exists(self.extends):
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


class PythonLanguage(gd.Class, inherits=classdb.ScriptLanguageExtension):
    def __init__(self):
        if hasattr(self.__class__, 'singleton'):
            raise TypeError("Python language singleton is already instantiated")

        self.__class__.singleton = self
        self._constants = {}
        self._scripts = {}

    @classmethod
    def get_singleton(cls):
        return cls.singleton

    def add_script(self, script):
        if not script.import_path or not script.module:
            raise TypeError("Script %r must be loaded" % script)
        self._scripts[script.import_path] =  script

    def _init(self):
        pass

    def _finish(self):
        pass

    def _get_name(self):
        return 'Python'

    def _get_type(self):
        return 'PythonScript'

    def _get_extension(self):
        return 'py'

    def _get_recognized_extensions(self) -> tuple:
        return ('py',)

    def _get_reserved_words(self):
        return (
            'False',
            'None',
            'True',
            'and',
            'as',
            'assert',
            'async',
            'await',
            'break',
            'case',
            'class',
            'continue',
            'def',
            'del',
            'elif',
            'else',
            'except',
            'finally',
            'for',
            'from',
            'global',
            'if',
            'import',
            'in',
            'is',
            'lambda',
            'match',
            'nonlocal',
            'not',
            'or',
            'pass',
            'raise',
            'return',
            'try',
            'type',
            'while',
            'with',
            'yield',
        )

    def _is_control_flow_keyword(self, keyword):
        return keyword in (
            'break',
            'case',
            'continue',
            'elif',
            'else',
            'except',
            'finally',
            'for',
            'if',
            'match',
            'pass',
            'raise',
            'return',
            'try',
            'while',
            'with',
            'yield'
        )


    def _get_comment_delimiters(self):
        return ('#',)

    def _get_string_delimiters(self):
        return ('"', "'")

    def _get_doc_comment_delimiters(self):
        return ('"""', "'''")


    def _make_template(self, template: str, class_name: str, base_class_name: str):
        template = template.replace('{CLASSNAME}', class_name).replace('{INHERITS}', base_class_name)
        res = PythonScript()
        res._set_source_code(template)

        return res


    def _get_built_in_templates(self, inherits: str) -> list:
        return [
            {
                'inherits': inherits,
                'name': 'Example Template',
                'description': 'This is an example template to show how to start',
                'content': 'class Example:\n    pass',  # TODO
                'origin': 1,
                'id': 1
            }
        ]

    def _is_using_templates(self):
        return True


    def _validate(self, code, path, validate_functions: bool, validate_errors: bool, validate_warnings: bool,
                  validate_safe_lines: bool):
        print("Validate %r from %r, %r, %r, %r, %r" %
              (code[:10] + '...', path, validate_functions, validate_errors, validate_warnings, validate_safe_lines))

        # TODO

        return {'valid': True}

    def _create_script(self):
        return PythonScript()

    def _has_named_classes(self):
        return True

    def _supports_builtin_mode(self):
        return False

    def _supports_documentation(self):
        return False

    def _can_inherit_from_file(self):
        return True

    def _preferred_file_name_casing(self):
        return 0

    def _can_make_function(self):
        return True

    def _find_function(self, class_name, function_name) -> int:
        return -1

    def _make_function(self, class_name, function_name, args: tuple) -> str:
        return '''\
    def %s%r:
        pass
    ''' % (function_name, args)

    def _lookup_code(self, s1, s2, s3, res):
        print("_lookup_code called with %r %r %r %r" % (s1, s2, s3, res))
        return {}

    def _overrides_external_editor(self):
        return False

    def _add_global_constant(self, name, value):
        self._constants[name] = value

    def _add_named_global_constant(self, name, value):
        self._constants[name] = value

    def _remove_named_global_constant(self, name, value):
        del self._constants[name]

    def _reload_all_scripts(self):
        for script in self._scripts.values():
            script.load()

    def _get_global_class_name(self, path) -> dict:
        return {}

    def _handles_global_class_type(self, path):
        return False


class ResourceFormatLoaderPythonScript(gd.Class, inherits=classdb.ResourceFormatLoader):
    def _get_recognized_extensions(self) -> tuple:
        return ('py',)

    def _handles_type(self, type: str) -> bool:
        return type == 'PythonScript'

    def _get_resource_type(self, path: str) -> str:
        if path.startswith('res://lib') or path == 'res://register_types.py':
            return False
        return 'PythonScript' if path.endswith('.py') else ''

    def _load(self, path: str, original_path: str, use_sub_threads: bool, cache_mode: int) -> gd.Extension:
        print('LOAD', path, original_path, use_sub_threads, cache_mode)
        res = PythonScript()
        res.load(path)

        print("Resource %r loaded. Returning it to engine" % res)
        return res


class ResourceFormatSaverPythonScript(gd.Class, inherits=classdb.ResourceFormatSaver):
    def _get_recognized_extensions(self, res) -> tuple:
        return ('py',)

    def _recognize(self, res):
        print("'_recognize' call", res)

        return False

    def _save(self, res, path, flags) -> gd.Error:
        print("'_save' call", res, path, flags)

        return res.save(path)
