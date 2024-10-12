import godot as gd


# PythonScript = gd.ExtensionClass('PythonScript', 'Resource')

# @PythonScript.bind_python_method
# def __init__(self):
#     self.source = ''


ResourceFormatLoaderPythonScript = gd.ExtensionClass('ResourceFormatLoaderPythonScript', 'ResourceFormatLoader')

def _get_recognized_extensions(self) -> tuple:
    # print('GET EXT')
    return ('py',)

def _handles_type(self, type: str) -> bool:
    # print("HANDLES TYPE", type)
    return type == 'PythonScript'

def _get_resource_type(self, path: str) -> str:
    print("GET RESOURCE TYPE", path)
    return 'PythonScript' if path.endswith('.py') else ''

def _load(self, path: str, original_path: str, use_sub_threads: bool, cache_mode: int) -> gd.Extension:
    print('LOAD', path, original_path, use_sub_threads, cache_mode)
    # res = PythonScript()
    return

ResourceFormatLoaderPythonScript.bind_virtual_methods(
    _get_recognized_extensions,
    _handles_type,
    _get_resource_type,
    _load,
)
