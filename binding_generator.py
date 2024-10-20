
#!/usr/bin/env python

import json
import gzip
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

    cython_gen_folder = Path(output_dir) / "gen" / "gdextension_interface"

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


def generate_api_data(api, output_dir):
    gen_folder = Path(output_dir) / "gdextension_interface"
    gen_folder.mkdir(parents=True, exist_ok=True)

    filename = gen_folder / "api_data.pxi"   
    ref_filename = gen_folder / "api_data_reference.py"

    api_data, reference = generate_api_data_file(
            api['classes'],
            api['builtin_classes'],
            api['singletons'],
            api['utility_functions'],
            api['global_enums'],
            api['native_structures']
    )

    print('Writing', filename)

    with filename.open("w+", encoding="utf-8") as file:
        file.write(api_data)

    with ref_filename.open("w+", encoding="utf-8") as file:
        file.write(reference)


def generate_api_data_file(classes, builtin_classes, singletons, utility_functions, enums, structs):
    singleton_data = set()
    inheritance_data = {}
    api_type_data = {}
    api_type_data_pickled = {}

    builtin_class_method_data = {}
    builtin_class_method_data_pickled = {}

    class_method_data = {}
    class_method_data_pickled = {}
    class_enum_data = {}
    class_enum_data_pickled = {}
    class_constant_data = {}
    class_constant_data_pickled = {}

    utilfunc_data = {}
    enum_data = {}
    struct_data = {}

    for singleton in singletons:
         singleton_data.add(singleton['name'])

    singleton_data_pickled = pickle.dumps(singleton_data)    

    for enum in enums:
        enum_data[enum['name']] = []
        for value in enum['values']:
            enum_data[enum['name']].append((value['name'], value['value']))

    enum_data_pickled = pickle.dumps(enum_data)

    for struct in structs:
        struct_data[struct['name']] = struct['format']

    struct_data_pickled = pickle.dumps(struct_data)

    for class_api in builtin_classes:
        class_name = class_api['name']

        if "methods" in class_api:
            builtin_class_method_data[class_name] = {}

            for method in class_api['methods']:
                method_name = method['name']
                type_info = []
                return_type = method.get('return_value', {}).get('type', 'Nil')
                type_info.append(return_type)
                method_info = {}
                for key in ('is_vararg', 'is_static', 'hash'):
                    method_info[key] = method[key]
                arguments = []
                if "arguments" in method:
                    for argument in method["arguments"]:
                        arg_name = argument['name']
                        arg_type = argument['type']
                        type_info.append(arg_type)
                        arguments.append((arg_name, arg_type))
                method_info['arguments'] = arguments
                method_info['type_info'] = tuple(type_info)
                builtin_class_method_data[class_name][method_name] = method_info

            builtin_class_method_data_pickled[class_name] = pickle.dumps(builtin_class_method_data[class_name])

    for class_api in classes:
        class_name = class_api['name']
        inheritance_data[class_name] = class_api.get('inherits', None)
        api_type_data.setdefault(class_api['api_type'], set()).add(class_name)

        if 'constants' in class_api:
            class_constant_data[class_name] = {}

            for constant in class_api['constants']:
                class_constant_data[class_name][constant['name']] = constant['value']
            
            class_constant_data_pickled[class_name] = pickle.dumps(class_constant_data[class_name])

        if 'enums' in class_api:
            class_enum_data[class_name] = {}

            for enum in class_api['enums']:
                enum_info = []
                enum_name = enum['name']
                for value in enum['values']:
                    enum_info.append((value['name'], value['value']))
                class_enum_data[class_name][enum_name] = enum_info
            
            class_enum_data_pickled[class_name] = pickle.dumps(class_enum_data[class_name])

        if "methods" in class_api:
            class_method_data[class_name] = {}

            for method in class_api['methods']:
                hash = method.get('hash', None)
                method_name = method['name']
                type_info = []
                return_type = method.get('return_value', {}).get('type', 'Nil')
                type_info.append(return_type)
                method_info = {
                    'hash': hash
                }
                for key in ('is_vararg', 'is_static', 'is_virtual'):
                    method_info[key] = method[key]
                arguments = []
                if "arguments" in method:
                    for argument in method["arguments"]:
                        arg_name = argument['name']
                        arg_type = argument['type']
                        type_info.append(arg_type)
                        arguments.append((arg_name, arg_type))
                method_info['arguments'] = arguments
                method_info['type_info'] = tuple(type_info)
                class_method_data[class_name][method_name] = method_info
        
            class_method_data_pickled[class_name] = pickle.dumps(class_method_data[class_name])

    for method in utility_functions:
        method_name = method['name']
        type_info = []
        return_type = method.get('return_value', {}).get('type', 'Nil')
        type_info.append(return_type)
        method_info = {
            'hash': method['hash']
        }
        arguments = []
        if "arguments" in method:
            for argument in method["arguments"]:
                arg_name = argument['name']
                arg_type = argument['type']
                type_info.append(arg_type)
                arguments.append({ arg_name: arg_type })
        method_info['arguments'] = arguments
        method_info['type_info'] = tuple(type_info)
        utilfunc_data[method_name] = method_info

    utilfunc_data_pickled = pickle.dumps(utilfunc_data)
    inheritance_data_pickled = pickle.dumps(inheritance_data)

    for (key, value) in api_type_data.items():
        api_type_data_pickled[key] = pickle.dumps(value)

    result = [
        'cdef bytes _global_singleton_info__pickle = \\\n%s' % pprint.pformat(singleton_data_pickled, width=120),
        'cdef bytes _global_enum_info__pickle = \\\n%s' % pprint.pformat(enum_data_pickled, width=120),
        'cdef bytes _global_struct_info__pickle = \\\n%s' % pprint.pformat(struct_data_pickled, width=120),
        'cdef bytes _global_inheritance_info__pickle = \\\n%s' % pprint.pformat(inheritance_data_pickled, width=120),
        'cdef dict _global_api_types__pickles = \\\n%s' % pprint.pformat(api_type_data_pickled, width=120),
        'cdef bytes _global_utility_function_info__pickle = \\\n%s' % pprint.pformat(utilfunc_data_pickled, width=120),

        'cdef dict _global_method_info__pickles = \\\n%s' % pprint.pformat(class_method_data_pickled, width=120),
        'cdef dict _global_class_enum_info__pickles = \\\n%s' % pprint.pformat(class_enum_data_pickled, width=120),
        'cdef dict _global_class_constant_info__pickles = \\\n%s' % pprint.pformat(class_constant_data_pickled, width=120),

        'cdef dict _global_builtin_method_info__pickles = \\\n%s' % pprint.pformat(builtin_class_method_data_pickled, width=120),
    ]

    result2 = [
        # '_global_type_info = \\\n%s' % pprint.pformat(type_data, width=120),
        '_global_singleton_info = \\\n%s' % pprint.pformat(singleton_data, width=120),
        '_global_enum_info = \\\n%s' % pprint.pformat(enum_data, width=120),
        '_global_inheritance_info = \\\n%s' % pprint.pformat(inheritance_data, width=120),
        #'_global_api_types = \\\n%s' % pprint.pformat(api_type_data, width=120),
        '_global_utility_function_info = \\\n%s' % pprint.pformat(utilfunc_data, width=120),

        '_global_method_info = \\\n%s' % pprint.pformat(class_method_data, width=120),
        '_global_class_enum_info = \\\n%s' % pprint.pformat(class_enum_data, width=120),
        '_global_class_constant_info = \\\n%s' % pprint.pformat(class_constant_data, width=120),

        '_global_builtin_method_info = \\\n%s' % pprint.pformat(builtin_class_method_data, width=120),
    ]

    return '\n\n'.join(result) + '\n', '\n\n'.join(result2) + '\n'
