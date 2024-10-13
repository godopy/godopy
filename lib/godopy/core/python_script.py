import godot as gd
from godot import classdb, types as gdtypes


class PythonScript(classdb.Resource):
    pass



class ResourceFormatLoaderPythonScript(classdb.ResourceFormatLoader):
    def _get_recognized_extensions(self) -> tuple:
        return ('py',)

    def _handles_type(self, type: str) -> bool:
        return type == 'PythonScript'

    def _get_resource_type(self, path: str) -> str:
        return 'PythonScript' if path.endswith('.py') else ''

    # def _recognize_path(self, path: str, hint: str) -> bool:
    #     return path.endswith('.py')

    # def _exists(self, path: str) -> bool:
    #     return path.endswith('.py')

    def _load(self, path: str, original_path: str, use_sub_threads: bool, cache_mode: int) -> gd.Extension:
        print('LOAD', path, original_path, use_sub_threads, cache_mode)
        res = classdb.Resource()

        res.ref_get_object()

        print(res, res.owner_id(), res.get_rid())

        print("Returning", res.get_reference_count())
        return res
