#ifndef GODOT_PYTHON__ICALLS_HPP
#define GODOT_PYTHON__ICALLS_HPP

#include <gdnative_api_struct.gen.h>
#include <stdint.h>

#include <pycore/PythonGlobal.hpp>
#include <core/CoreTypes.hpp>
<%!
  from pygodot.bindings._generator import is_class_type, get_icall_return_type
%>
% for ret, args, sig, nonvoid in icalls:
static inline ${sig} {
  % if nonvoid:
    ${'godot_object *' if is_class_type(ret) else get_icall_return_type(ret)} ret;
    % if is_class_type(ret):
    ret = nullptr;
    % endif
  % endif
  const void *args[${'' if args else '1'}] = {
  % for i, arg in enumerate(args):
    % if is_class_type(arg):
    (void *) arg${i},
    % else:
    (void *) &arg${i},
    % endif
  % endfor
  };
  godot::api->godot_method_bind_ptrcall(mb, o, args, ${'&ret' if nonvoid else 'nullptr'});
  % if nonvoid:
    % if is_class_type(ret):
  if (ret) return (__pygodot___Wrapped *)godot::nativescript_1_1_api->godot_nativescript_get_instance_binding_data(godot::_RegisterState::python_language_index, ret);
  return (__pygodot___Wrapped *)ret;
    % else:
  return ret;
    % endif
  % endif
}
% endfor
#endif
