#ifndef GODOT_PYTHON__ICALLS_HPP
#define GODOT_PYTHON__ICALLS_HPP

#include <gdnative_api_struct.gen.h>
#include <stdint.h>

#include <pycore/PythonGlobal.hpp>
#include <core/CoreTypes.hpp>
<%!
    from pygodot.bindings._generator import is_class_type, get_icall_return_type

    def arg_elem(index, arg):
        t = '(void *)arg%d' if is_class_type(arg) else '(void *)&arg%d'
        return t % index
%>
namespace godot {
% for ret, args, sig, pxd_sig in icalls:

static inline ${sig} {
  % if ret != 'void':
  ${'godot_object *ret = NULL' if is_class_type(ret) else get_icall_return_type(ret) + 'ret'};
  % endif
  const void *args[${'' if args else '1'}] = {${', '.join(arg_elem(i, arg) for i, arg in enumerate(args))}};
  godot::api->godot_method_bind_ptrcall(mb, o, args, ${'&ret' if ret != 'void' else 'NULL'});
  % if ret != 'void':
    % if is_class_type(ret):
  if (ret)
    return (PyObject *)godot::nativescript_1_1_api->godot_nativescript_get_instance_binding_data(godot::_RegisterState::cython_language_index, ret);
  return (PyObject *)ret;
    % else:
  return ret;
    % endif
  % endif
}
% endfor
}
#endif
