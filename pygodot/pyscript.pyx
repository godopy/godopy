#cython: c_string_encoding=utf-8
from cython.operator cimport dereference as deref
from cpython.object cimport PyObject

from .headers.gdnative_api cimport *
from .globals cimport gdapi, _python_language_index
from .utils cimport godot_string_to_bytes

# Don't do relative pure-Python imports in Cython modules
from pygodot.utils import _pyprint as print

import os
from mako.template import Template

cdef PyScriptLanguage _language = None;
cdef object _pyscripts = set()

### PyScript "language" implementation ###

cdef class PyScriptLanguage:
    cdef str homedir

    def __cinit__(self):
        import pygodot
        self.homedir = os.path.realpath(os.path.dirname(pygodot.__file__))


cdef public godot_pluginscript_language_data *pyscript_language_init():
    global _language
    _language = PyScriptLanguage()

    print('PyScript init language', _language)
    return <godot_pluginscript_language_data *>_language


cdef public void pyscript_language_finish(godot_pluginscript_language_data *p_data):
    global _language
    print('PyScript finish language', _language)
    _language = None
    _pyscripts.clear() # No leaks, ma!

cdef public godot_string pyscript_get_template_source_code(godot_pluginscript_language_data *p_data,
                                                           const godot_string *p_class_name,
                                                           const godot_string *p_base_class_name):
    cdef object data = <PyScriptLanguage>p_data
    print('TEMPLATE SOURCE!', data)

    if not isinstance(data, PyScriptLanguage):
        raise RuntimeError('PyScript is not initialized properly. '
            'Make sure that PyGodot library is installed as a singleton.')

    cdef str class_name = godot_string_to_bytes(p_class_name).encode('utf-8')
    cdef str base_class_name = godot_string_to_bytes(p_base_class_name).encode('utf-8')
    template = Template(os.path.join(data.homedir, 'templates', 'source_code.mako'))

    cdef bytes ret = template.render(class_name=class_name, base_class_name=base_class_name).encode('utf-8')

    return godot_string_chars_to_utf8_with_len(<const char *>ret, len(ret))

# TODO: validate, find_function, make_function, complete_code, auto_indent_code


cdef public void pyscript_add_global_constant(godot_pluginscript_language_data *p_data, const godot_string *p_variable,
                                              const godot_variant *p_value):
    name = godot_string_to_bytes(p_variable)
    print('pyscript_add_global_constant', name)

# TODO: debug_get_error, etc.

# TODO: public_functions, public_constants

# TODO: profiling_start, etc.


### PyScript "script" implementation ###


import re
import importlib

cdef class PyScript:
    cdef godot_pluginscript_script_manifest manifest

    cdef bytes name
    cdef bytes base

    cdef object mod
    cdef object cls
    cdef object exception

    def __cinit__(self, bytes b_path):
        self.exception, self.mod, self.cls = self._init_python_module(b_path)

        if self.cls is not None:
            self.name = self.cls.__name__.encode('utf-8')
            self.base = self.cls.__bases__[0].__name__.encode('utf-8')
            self.cls.__godot_path__ = b_path.decode('utf-8')
        else:
            self.name = self.base = b'Error'

    @staticmethod
    def _init_python_module(path):
        import_path = re.sub(r'^res\:\/\/', 'demo.', path.decode('utf-8'))
        import_path = re.sub(r'\.pyw?$', '', import_path.replace('/', '.'))
        print('initscript', import_path)

        class_name = import_path.split('.').pop()

        try:
            mod = importlib.import_module(import_path)
        except ImportError as ex:
            print(ex)
            return ex, None, None
        except Exception as ex:
            print(ex)
            return ex, None, None

        # TODO: Allow to work at module level as GDScript
        try:
            cls = getattr(mod, class_name)
        except AttributeError as ex:
            print(ex)
            return ex, mod, None

        return None, mod, cls


cdef public godot_pluginscript_script_manifest pyscript_init(godot_pluginscript_language_data *p_data,
                                                             const godot_string *p_path, const godot_string *p_source,
                                                             godot_error *r_error):
    cdef PyScript script = PyScript(godot_string_to_bytes(p_path))

    print('PyScript init script', script)

    # print('PyScript source', source.decode('utf-8'))

    script.manifest.data = <godot_pluginscript_script_data *><PyObject *>script

    godot_string_name_new_data(&script.manifest.name, <const char *>script.name)
    godot_string_name_new_data(&script.manifest.base, <const char *>script.base)

    script.manifest.is_tool = False

    godot_dictionary_new(&script.manifest.member_lines)
    godot_array_new(&script.manifest.methods)
    godot_array_new(&script.manifest.signals)
    godot_array_new(&script.manifest.properties)

    _pyscripts.add(script)

    return script.manifest


# XXX: may not be called!
cdef public void pyscript_finish(godot_pluginscript_script_data *p_script):
    cdef object script = <PyScript>p_script
    print('PyScript finish', script)
    if script in _pyscripts:
        _pyscripts.remove(script)

# cdef public void pyscript_get_name(godot_pluginscript_script_data *p_script, godot_string *r_name):
#     pass

# cdef public godot_bool pyscript_is_tool(godot_pluginscript_script_data *p_script):
#     return False

# cdef public godot_bool pyscript_can_instance(godot_pluginscript_script_data *p_script):
#     return False

### Pyscript "instance" implementation

cdef public godot_pluginscript_instance_data *pyscript_instance_init(godot_pluginscript_script_data *p_script,
                                                                     godot_object *obj):
    print(<object>p_script, 'INSTANCE REQUESTED')
    return NULL

cdef public void pyscript_instance_finish(godot_pluginscript_instance_data *p_instance):
    pass

cdef public void pyscript_instance_refcount_incremented(godot_pluginscript_instance_data *p_instance):
    pass

cdef public bool pyscript_instance_refcount_decremented(godot_pluginscript_instance_data *p_instance):
    # Return true if it can die
    return False

cdef public godot_bool pyscript_instance_set_prop(godot_pluginscript_instance_data *p_instance,
                                                  const godot_string *p_name, const godot_variant *p_value):
    return False

cdef public godot_bool pyscript_instance_get_prop(godot_pluginscript_instance_data *p_instance,
                                                  const godot_string *p_name, godot_variant *r_ret):
    return False

cdef public godot_variant pyscript_instance_call_method(godot_pluginscript_instance_data *p_instance,
                                                        const godot_string_name *p_method,
                                                        const godot_variant **p_args, int p_argcount,
                                                        godot_variant_call_error *r_error):
    cdef godot_variant ret
    gdapi.godot_variant_new_nil(&ret)
    return ret

cdef public void pyscript_instance_notification(godot_pluginscript_instance_data *handle, int notification):
    pass

# TODO: get_rpc_mode, get_rset_mode
