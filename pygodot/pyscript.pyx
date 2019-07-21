#cython: c_string_encoding=utf-8
from cython.operator cimport dereference as deref
from cpython.object cimport PyObject

from .headers.gdnative_api cimport *
from .globals cimport gdapi, _python_language_index
from ._core cimport _Wrapped
from .utils cimport godot_string_to_bytes, godot_project_dir

# Don't do relative pure-Python imports in Cython modules
from pygodot.utils import _pyprint as print

import re
import os
import importlib
import inspect
from mako.template import Template

### PyScript "language" implementation

cdef PyScriptLanguage _language;

cdef class PyScriptLanguage:
    cdef str templates_dir
    cdef str godot_project_dir
    cdef str resource_root
    cdef set scripts
    cdef bint nativescript_ready

    def __cinit__(self):
        # NOTES:
        # - This is called before nativescript_init and after gdnative_singleton — no normal binding calls here
        # - Direct calls to `method_bind_get_method`/`method_bind_ptrcall` should probably work everywhere
        # - If gdblib is not marked as a singleton in the Project settings, this will never be called and the language
        #   will not be installed properly (ScriptServer initialization happens quite early)
        # - If no gdns resources are installed, NS would not initialize
        # - Check for autoloaded PyGodotGlobal ensures NS initialization (which registers base Python classes)
        import pygodot
        # TODO: Use pre-compiled templates everywhere as they are more portable
        pygodot_package_dir = os.path.realpath(os.path.dirname(pygodot.__file__)) # XXX may not work in the frozen env
        self.templates_dir = os.path.join(pygodot_package_dir, 'templates')
        self.godot_project_dir = godot_project_dir()
        self.resource_root = os.path.basename(self.godot_project_dir)

        self.nativescript_ready = False
        # Maintain Python reference counts, because references get lost as soon as they leave Python
        self.scripts = set()

    def __repr__(self):
        return '<PyScriptLanguage 0x%x, resource_root=%s>' % (hash(self), self.resource_root)

    cpdef godot_path_to_qualified_name(self, str path):
        # https://docs.python.org/3/glossary.html#term-qualified-name
        # From PyGodot's POV a Godot project is just another Python package
        parent_path = re.sub(r'^res\:\/\/', self.resource_root + '.', path)  # use urlparse
        name, ext = os.path.splitext(parent_path)
        return name.replace('/', '.')

cdef public godot_pluginscript_language_data *pyscript_language_init():
    global _language
    _language = PyScriptLanguage()

    print('[PyPS] PyScript init language', _language)
    return <godot_pluginscript_language_data *>_language


cdef public void pyscript_language_finish(godot_pluginscript_language_data *p_data):
    global _language
    print('[PyPS] PyScript finish language', _language)


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
    template = Template(os.path.join(data.templates_dir, 'source_code.mako'))

    cdef bytes ret = template.render(class_name=class_name, base_class_name=base_class_name).encode('utf-8')

    return godot_string_chars_to_utf8_with_len(<const char *>ret, len(ret))

# TODO: validate, find_function, make_function, complete_code, auto_indent_code


cdef public void pyscript_add_global_constant(godot_pluginscript_language_data *p_data, const godot_string *p_variable,
                                              const godot_variant *p_value):
    language = <PyScriptLanguage>p_data
    if godot_string_to_bytes(p_variable) == b'PyGodotGlobal':
        language.nativescript_ready = True


# TODO: debug_get_error, etc.

# TODO: public_functions, public_constants

# TODO: profiling_start, etc.


### PyScript "script" implementation

cdef inline godot_variant *str_to_variant(str s):
    cdef godot_variant v
    cdef bytes b = s.encode('utf-8')
    cdef godot_string gd_s
    gdapi.godot_string_new(&gd_s)
    gdapi.godot_string_parse_utf8(&gd_s, <const char *>b)
    godot_variant_new_string(&v, &gd_s)
    # gdapi.godot_string_destroy(&gd_s)
    return &v

cdef inline godot_variant *nil():
    cdef godot_variant ret
    gdapi.godot_variant_new_nil(&ret)
    return &ret


cdef class PyScript:
    cdef godot_pluginscript_script_manifest manifest

    cdef bytes name
    cdef bytes base

    cdef object mod
    cdef object cls
    cdef godot_error error

    cdef set instances

    cdef list _godot_dictionary_allocations

    def __cinit__(self, language, bytes b_path):
        self.error, self.mod, self.cls = self._init_python_module(language, b_path)

        if self.cls is not None:
            self.name = self.cls.__name__.encode('utf-8')
            self.base = self.cls.__bases__[0].__name__.encode('utf-8')
            self.cls.__godot_path__ = b_path.decode('utf-8')
            self.cls.__pygodot_script__ = self
        else:
            self.name = b'Error'
            self.base = b''

        self._godot_dictionary_allocations = []
        self.manifest.data = <godot_pluginscript_script_data *><PyObject *>self

        godot_string_name_new_data(&self.manifest.name, <const char *>self.name)
        godot_string_name_new_data(&self.manifest.base, <const char *>self.base)

        self.manifest.is_tool = False

        self._load_members()
        self._load_methods()
        self._load_signals()
        self._load_properties()

        # Maintain Python reference counts, because references get lost as soon as they leave Python
        self.instances = set()

    def _load_members(self):
        godot_dictionary_new(&self.manifest.member_lines)

    def _load_methods(self):
        godot_array_new(&self.manifest.methods)

    def _load_signals(self):
        godot_array_new(&self.manifest.signals)

    def _load_properties(self):
        godot_array_new(&self.manifest.properties)

    def _init_python_module(self, language, path):
        import_path = language.godot_path_to_qualified_name(path.decode('utf-8'))

        class_name = import_path.split('.').pop()

        try:
            mod = importlib.import_module(import_path)
        except ImportError as ex:
            print(ex)
            return GODOT_ERR_CANT_OPEN, None, None
        except Exception as ex:
            print(ex)
            return GODOT_ERR_BUG, None, None

        # TODO: Allow to work at module level as GDScript
        try:
            cls = getattr(mod, class_name)
        except AttributeError as ex:
            print(ex)
            return GODOT_ERR_METHOD_NOT_FOUND, mod, None

        return GODOT_OK, mod, cls

cdef public godot_pluginscript_script_manifest pyscript_init(godot_pluginscript_language_data *p_data,
                                                             const godot_string *p_path, const godot_string *p_source,
                                                             godot_error *r_error):
    language = <PyScriptLanguage>p_data

    cdef PyScript script = PyScript(language, godot_string_to_bytes(p_path))

    if script.error != GODOT_OK:
        r_error[0] = script.error
        return script.manifest

    if not language.nativescript_ready:
        print("Base Godot classes are not loaded! Please, add the PyGodotGlobal nativescript to the \"AutoLoad\" list.")
        r_error[0] = GODOT_ERR_BUG
        return script.manifest

    if not issubclass(script.cls, _Wrapped):
        print("Python class does not inherit from proper PyGodot base classes")
        r_error[0] = GODOT_ERR_BUG
        return script.manifest

    language.scripts.add(script)

    r_error[0] = GODOT_OK
    print('[PyNS] PyScript init script', script, hex(<size_t>&script.manifest))
    return script.manifest


# Won't be called if pyscript_init wouldn't set r_error to GODOT_OK
cdef public void pyscript_finish(godot_pluginscript_script_data *p_script):
    cdef object script = <PyScript>p_script
    print('PyScript finish', script)



### Pyscript "instance" implementation

cdef public godot_pluginscript_instance_data *pyscript_instance_init(godot_pluginscript_script_data *p_script,
                                                                     godot_object *obj):
    script = <PyScript>p_script

    instance = script.cls()
    script.instances.add(instance)
    return <godot_pluginscript_instance_data *><PyObject *>instance

cdef public void pyscript_instance_finish(godot_pluginscript_instance_data *p_instance):
    instance = <object>p_instance
    instance.__class__.__pygodot_script__.instances.remove(instance)

# TODO
cdef public void pyscript_instance_refcount_incremented(godot_pluginscript_instance_data *p_instance):
    pass

# TODO
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
    instance = <object>p_instance
    cdef godot_string gd_method = gdapi.godot_string_name_get_name(p_method)
    method = godot_string_to_bytes(&gd_method).decode('utf-8')

    if hasattr(instance, method):
        python_method = getattr(instance, method)
        try:
            python_method() # TODO: args and return values
        except Exception as ex:
            print('METHOD CALL ERROR:', ex)
            r_error.error = GODOT_CALL_ERROR_CALL_ERROR_INVALID_METHOD
        else:
            r_error.error = GODOT_CALL_ERROR_CALL_OK
    else:
        print('method "%s" not found' % method)
        r_error.error = GODOT_CALL_ERROR_CALL_ERROR_INVALID_METHOD

    cdef godot_variant ret
    gdapi.godot_variant_new_nil(&ret)
    return ret

cdef public void pyscript_instance_notification(godot_pluginscript_instance_data *handle, int notification):
    pass

# TODO: get_rpc_mode, get_rset_mode
