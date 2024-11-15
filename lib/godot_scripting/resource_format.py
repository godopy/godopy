from typing import List

import godot
from godot.classdb import ResourceFormatLoader, ResourceFormatSaver, Resource

from .script import PythonScript


class ResourceFormatLoaderPython(godot.Class, inherits=ResourceFormatLoader, no_virtual_underscore=True):
    def get_recognized_extensions(self) -> List[str]:
        return ['py']

    def handles_type(self, type: str) -> bool:
        return type == 'Script' or type == 'PythonScript'

    def get_resource_type(self, path: str) -> str:
        if path.startswith('res://lib') or path == 'res://register_types.py':
            return False
        return 'Python' if path.endswith('.py') else ''

    def load(self, path: str, original_path: str, use_sub_threads: bool, cache_mode: int) -> PythonScript:
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
        return res._save(path)
