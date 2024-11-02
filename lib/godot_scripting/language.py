
from typing import List

import godot
from godot.classdb import ScriptLanguageExtension

from .script import Python


class PythonLanguage(godot.Class, inherits=ScriptLanguageExtension):
    def __init__(self):
        if hasattr(self.__class__, 'singleton'):
            raise TypeError("Python language singleton already exists")

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
        return 'Python'

    def _get_extension(self):
        return 'py'

    def _get_recognized_extensions(self) -> List[str]:
        return ['py']

    def _get_reserved_words(self) -> List[str]:
        return [
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
        ]

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
        return ['#']

    def _get_string_delimiters(self):
        return ['"', "'"]

    def _get_doc_comment_delimiters(self):
        return ['"""', "'''"]


    def _make_template(self, template: str, class_name: str, base_class_name: str):
        template = template.replace('{CLASSNAME}', class_name).replace('{INHERITS}', base_class_name)
        res = Python()
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
        return Python()

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
