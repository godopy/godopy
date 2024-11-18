from typing import List

import godot
from godot.classdb import ResourceFormatLoader, ResourceFormatSaver, Resource

from .script import PythonScript, PythonProjectSettings, PythonExtension


class ResourceFormatLoaderPython(godot.Class, inherits=ResourceFormatLoader, no_virtual_underscore=True):
    def get_recognized_extensions(self) -> List[str]:
        return ['py']


    def handles_type(self, type: str) -> bool:
        return type == 'Script' or type == 'Python' or type == 'PythonProjectSettings'


    def get_resource_type(self, path: str) -> str:
        if path.startswith('res://lib') or path == 'res://register_types.py':
            return 'PythonExtension' if path.endswith('.py') else ''

        if path == 'res://project.py':
            return 'PythonProjectSettings'

        return 'PythonScript' if path.endswith('.py') else ''


    def load(self, path: str, original_path: str, use_sub_threads: bool, cache_mode: int) -> PythonScript:
        if path == 'res://project.py':
            res = PythonProjectSettings()

            return res

        elif path.startswith('res://lib') or path == 'res://register_types.py':
            res = PythonExtension()

            return res

        else:
            res = PythonScript()
            res._load(path)

            return res


class ResourceFormatSaverPython(godot.Class, inherits=ResourceFormatSaver, no_virtual_underscore=True):
    def get_recognized_extensions(self, res: Resource) -> List[str]:
        if isinstance(res, PythonScript):
            return ['py']
        return []

    def recognize(self, res) -> bool:
        return isinstance(res, PythonScript)

    def save(self, res, path, flags) -> godot.Error:
        if isinstance(res, PythonScript):
            return res._save(path)
