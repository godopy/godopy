
from typing import List

import godot
from godot.classdb import ScriptLanguageExtension

from .script import PythonScript


class PythonLanguage(godot.Class, inherits=ScriptLanguageExtension, no_virtual_underscore=True):
    def __init__(self):
        if hasattr(self.__class__, 'singleton'):
            raise ValueError("Python language singleton already exists")

        self.__class__.singleton = self
        self._constants = {}
        self._scripts = {}

    def _add_script(self, script):
        if not script._import_path or not script._module:
            raise TypeError("Script %r must be loaded" % script)
        self._scripts[script._import_path] =  script

    def _init(self):
        pass

    def _finish(self):
        pass

    def get_name(self):
        return 'Python'

    def get_type(self):
        return 'Python'

    def get_extension(self):
        return 'py'

    def get_recognized_extensions(self) -> List[str]:
        return ['py']

    def get_reserved_words(self) -> List[str]:
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

    def is_control_flow_keyword(self, keyword):
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


    def get_comment_delimiters(self):
        return ['#']

    def get_string_delimiters(self):
        return ['"', "'"]

    def get_doc_comment_delimiters(self):
        return ['"""', "'''"]


    def make_template(self, template: str, class_name: str, base_class_name: str):
        template = template.replace('{CLASSNAME}', class_name).replace('{INHERITS}', base_class_name)
        res = PythonScript()
        res.set_source_code(template)

        return res


    def get_built_in_templates(self, inherits: str) -> list:
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

    def is_using_templates(self):
        return True


    def validate(self, code, path, validate_functions: bool, validate_errors: bool, validate_warnings: bool,
                  validate_safe_lines: bool):
        print("Validate %r from %r, %r, %r, %r, %r" %
              (code[:10] + '...', path, validate_functions, validate_errors, validate_warnings, validate_safe_lines))

        # TODO

        return {'valid': True}

    def create_script(self):
        print("create_script")
        return PythonScript()

    def has_named_classes(self):
        return True

    def supports_builtin_mode(self):
        return False

    def supports_documentation(self):
        return False

    def can_inherit_from_file(self):
        return True

    def preferred_file_name_casing(self):
        return 0

    def can_make_function(self):
        return True

    def find_function(self, class_name, function_name) -> int:
        print('find_function', class_name, function_name)
        return -1

    def make_function(self, class_name, function_name, args: tuple) -> str:
        print("make_function", )
        return '''\
    def %s%r:
        pass
    ''' % (function_name, args)

    def lookup_code(self, s1, s2, s3, res):
        print("lookup_code called with %r %r %r %r" % (s1, s2, s3, res))
        return {}

    def overrides_external_editor(self):
        return False

    def add_global_constant(self, name, value):
        print("add_global_constant", name, value)
        self._constants[name] = value

    def add_named_global_constant(self, name, value):
        print("add_named_global_constant", name, value)
        self._constants[name] = value

    def _remove_named_global_constant(self, name, value):
        del self._constants[name]

    def _reload_all_scripts(self):
        for script in self._scripts.values():
            script.load()

    def _get_global_class_name(self, path) -> dict:
        print("get_global_class_name", path)
        return {}

    def _handles_global_class_type(self, res):
        return res == 'Python'
