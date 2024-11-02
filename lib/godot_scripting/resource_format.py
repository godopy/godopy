from typing import List

import godot
from godot.classdb import ResourceFormatLoader, ResourceFormatSaver

from .script import Python


class ResourceFormatLoaderPython(godot.Class, inherits=ResourceFormatLoader):
    def _get_recognized_extensions(self) -> List[str]:
        return ['py']

    def _handles_type(self, type: str) -> bool:
        return type == 'Script' or type == 'PythonScript'

    def _get_resource_type(self, path: str) -> str:
        if path.startswith('res://lib') or path == 'res://register_types.py':
            return False

        return 'Python' if path.endswith('.py') else ''

    def _load(self, path: str, original_path: str, use_sub_threads: bool, cache_mode: int) -> Python:
        print('LOAD', path, original_path, use_sub_threads, cache_mode)
        res = Python()
        res.load(path)

        print("Resource %r loaded. Returning it to engine" % res)
        return res


class ResourceFormatSaverPython(godot.Class, inherits=ResourceFormatSaver):
    def _get_recognized_extensions(self, res) -> List[str]:
        return ['py']


    def _recognize(self, res) -> bool:
        print("'_recognize' call", res)

        return isinstance(res, Python)


    def _save(self, res, path, flags) -> godot.Error:
        print("'_save' call", res, path, flags)

        return res.save(path)
