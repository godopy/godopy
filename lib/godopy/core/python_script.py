import godot as gd
from godot import classdb


class PythonScript(classdb.Resource):
    def __init__(self):
        self.source = []


class ResourceFormatLoaderPythonScript(classdb.ResourceFormatLoader):
    def _get_recognized_extensions(self) -> tuple:
        return ('py',)

    def _handles_type(self, type: str) -> bool:
        return type == 'PythonScript'

    def _get_resource_type(self, path: str) -> str:
        return 'PythonScript' if path.endswith('.py') else ''

    def _load(self, path: str, original_path: str, use_sub_threads: bool, cache_mode: int) -> gd.Extension:
        print('LOAD', path, original_path, use_sub_threads, cache_mode)
        res = PythonScript()
        return res
