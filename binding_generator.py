
#!/usr/bin/env python

import json
import re
import shutil
from pathlib import Path
import pprint
import pickle


def scons_emit_files(target, source, env):
    files = [env.File(f) for f in get_file_list(target[0].abspath)]
    env.Clean(target, files)
    env["godopy_gen_dir"] = target[0].abspath
    return files, source


def scons_generate_bindings(target, source, env):
    generate_bindings(
        str(source[0]),
        env["godopy_gen_dir"],
    )


def get_file_list(output_dir):
    files = []

    cython_gen_folder = Path(output_dir) / "gen" / "cythonlib"

    files.append(str((cython_gen_folder / "api_data.pxi").as_posix()))

    return files


def generate_bindings(api_filepath, output_dir="."):
    api = None

    target_dir = Path(output_dir) / "gen"

    with open(api_filepath, encoding="utf-8") as api_file:
        api = json.load(api_file)

    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True)

    print('Generating API bindings...')

    generate_api_data(api, target_dir)

    # generate_global_constants(api, target_dir)
    # generate_version_header(api, target_dir)
    # generate_global_constant_binds(api, target_dir)
    # generate_builtin_bindings(api, target_dir, real_t + "_" + bits)
    # generate_engine_classes_bindings(api, target_dir, use_template_get_node)
    # generate_utility_functions(api, target_dir)

def generate_api_data(api, output_dir):
    gen_folder = Path(output_dir) / "cythonlib"
    gen_folder.mkdir(parents=True, exist_ok=True)

    filename = gen_folder / "api_data.pxi"   

    print('writing', filename)

    with filename.open("w+", encoding="utf-8") as file:
            file.write(
                generate_api_data_file(
                     api['classes'],
                     api['singletons'],
                     api['utility_functions']
                )
            )

def generate_api_data_file(classes, singletons, utility_functions):
    singleton_data = set()
    for singleton in singletons:
         singleton_data.add(singleton['name'])
    class_method_data = {}
    for class_api in classes:
        class_name = class_api['name']
        if "methods" in class_api:
            class_method_data[class_name]  = {'is_singleton': class_name in singletons}
            for method in class_api['methods']:
                hash = method.get('hash')
                if hash is None:
                    continue
                method_name = method['name']
                method_info = {
                    'return_type': method.get('return_value', {}).get('type', 'Nil'),
                    'hash': hash
                }
                # arguments = []
                # if "arguments" in method:
                #     for argument in method["arguments"]:
                #         arg_name = argument['name']
                #         arguments.append({ argument['name'], argument['type'] })
                # method_info['arguments'] = arguments
                class_method_data[class_name][method_name] = method_info
            class_method_data[class_name] = pickle.dumps(class_method_data[class_name])

    utilfunc_data = {}
    for method in utility_functions:
        method_name = method['name']
        method_info = {
            'return_type': method.get('return_value', {}).get('type', 'Nil'),
            'hash': method['hash']
        }
        utilfunc_data[method_name] = method_info
    utilfunc_data = pickle.dumps(utilfunc_data)

    result = [
        # 'cdef set _singleton_names = \\\n%s' % pprint.pformat(singleton_data, width=120),
        'cdef bytes _utility_function_data = \\\n%s' % pprint.pformat(utilfunc_data, width=120),
        'cdef dict _method_data = \\\n%s' % pprint.pformat(class_method_data, width=120),

    ]
    return '\n\n'.join(result) + '\n'

