
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

    files.append(str((cython_gen_folder / "api_data.pickle").as_posix()))

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

    filename = gen_folder / "api_data.pickle"
    ref_filename = gen_folder / "api_data_reference.py"

    api_data, reference = generate_api_data_file(
            api['header'],
            api['classes'],
            api['builtin_classes'],
            api['singletons'],
            api['utility_functions'],
            api['global_enums'],
            api['native_structures']
    )

    print('Writing', filename)

    with filename.open("wb") as file:
        file.write(api_data)

    with ref_filename.open("w", encoding="utf-8") as file:
        file.write(reference)


BUILTIN_TYPE_LIST = [
    'Nil',

    # atomic types
    'bool',
    'int',
    'float',
    'String',

    # math types
    'Vector2',
    'Vector2i',
    'Rect2',
    'Rect2i',
    'Vector3',
    'Vector3i',
    'Transform2D',
    'Vector4',
    'Vector4i',
    'Plane',
    'Quaternion',
    'AABB',
    'Basis',
    'Transform3D',
    'Projection',

    # misc types
    'Color',
    'StringName',
    'NodePath',
    'RID',
    'Object',
    'Callable',
    'Signal',
    'Dictionary',
    'Array',

    # typed arrays
    'PackedByteArray',
    'PackedInt32Array',
    'PackedInt64Array',
    'PackedFloat32Array',
    'PackedFloat64Array',
    'PackedStringArray',
    'PackedVector2Array',
    'PackedVector3Array',
    'PackedColorArray',
    'PackedVector4Array',
]

def generate_api_data_file(header, classes, builtin_classes, singletons, utility_functions, enums, structs):
    singleton_data = set()
    inheritance_data = {}
    api_type_data = {}

    builtin_class_method_data = {}

    class_method_data = {}
    class_enum_data = {}
    class_enum_data_pickled = {}
    class_constant_data = {}

    utilfunc_data = {}
    enum_data = {}
    struct_data = set()

    max_types = 0

    for singleton in singletons:
         singleton_data.add(singleton['name'])

    for enum in enums:
        enum_data[enum['name']] = []
        for value in enum['values']:
            enum_data[enum['name']].append((value['name'], value['value']))

    all_enums = set()

    all_enums |= set(enum_data.keys())

    for struct in structs:
        struct_data.add(struct['name'])

    for class_api in builtin_classes:
        class_name = class_api['name']

        if "methods" in class_api:
            builtin_class_method_data[class_name] = {}

            for method in class_api['methods']:
                method_name = method['name']
                type_info = []
                return_type = method.get('return_type', 'Nil')
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

    for class_api in classes:
        class_name = class_api['name']
        inheritance_data[class_name] = class_api.get('inherits', None)
        api_type_data.setdefault(class_api['api_type'], set()).add(class_name)

        if 'constants' in class_api:
            class_constant_data[class_name] = {}

            for constant in class_api['constants']:
                class_constant_data[class_name][constant['name']] = constant['value']

        if 'enums' in class_api:
            class_enum_data[class_name] = {}

            for enum in class_api['enums']:
                enum_info = []
                enum_name = enum['name']
                for value in enum['values']:
                    enum_info.append((value['name'], value['value']))
                class_enum_data[class_name][enum_name] = enum_info
            
            class_enum_data_pickled[class_name] = pickle.dumps(class_enum_data[class_name])
            all_enums |= set(class_enum_data[class_name].keys())

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
                if len(type_info) > max_types:
                    max_types = len(type_info)
                    longest_method_info = method_name, method_info
                class_method_data[class_name][method_name] = method_info

    for method in utility_functions:
        method_name = method['name']
        type_info = []
        return_type = method.get('return_value', {}).get('type', 'Nil')
        type_info.append(return_type)
        method_info = {
            'hash': method['hash'],
            'is_vararg': method['is_vararg']
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


    data = {
        'api_header': header,
        'global_singleton_info': singleton_data,
        'global_enum_info': enum_data,
        'global_struct_info': struct_data,
        'global_inheritance_info': inheritance_data,
        'global_utility_function_info': utilfunc_data,
        'global_method_info': class_method_data,
        'global_class_enum_info': class_enum_data,
        'global_class_constant_info': class_constant_data,
        'global_builtin_method_info': builtin_class_method_data
    }

    result = pickle.dumps(data)

    result2 = 'api_data_pickle = %s\n' % pprint.pformat(data, width=120)

    return result, result2
