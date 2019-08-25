#ifndef GODOT_PYTHON__ICALLS_HPP
#define GODOT_PYTHON__ICALLS_HPP

#include <gdnative_api_struct.gen.h>
#include <stdint.h>

#include <core/GodotGlobal.hpp>
#include <pycore/PythonGlobal.hpp>
#include <core/CoreTypes.hpp>
<%!
    from godot_tools.binding_generator import is_class_type, get_icall_return_type

    def arg_elem(index, arg):
        t = '(void *)arg%d' if is_class_type(arg) else '(void *)&arg%d'
        return t % index

    def given_arg_elem(index, arg):
        return '(Variant)arg%d' % index
%>
namespace godot {
% for ret, args, has_varargs, sig, pxd_sig in icalls:

static inline ${sig} {
  % if has_varargs:
  % if args:
  const Variant __given_args[] = {${', '.join(given_arg_elem(i, arg) for i, arg in enumerate(args))}};
  % endif

  int __size = __var_args.size();
  godot_variant **__args = (godot_variant **) godot::api->godot_alloc(sizeof(godot_variant *) * (__size + ${len(args)}));
  % for i, arg in enumerate(args):
  __args[${i}] = (godot_variant *) &__given_args[${i}];
  % endfor

  for (int i = 0; i < __size; i++) {
    __args[i + ${len(args)}] = (godot_variant *) &((Array &) __var_args)[i];
  }

  Variant __result;
  *(godot_variant *) &__result = godot::api->godot_method_bind_call(mb, o, (const godot_variant **) __args, __size + ${len(args)}, nullptr);
  % for i, arg in enumerate(args):
  godot::api->godot_variant_destroy((godot_variant *) &__given_args[${i}]);
  % endfor
  % if ret != 'void':
  return __result;
  % endif ## != void
  % else: ## no varargs
  % if ret != 'void':
  ${'PyObject *ret = NULL' if is_class_type(ret) else get_icall_return_type(ret) + 'ret'};
  % endif

  const void *args[${'' if args else '1'}] = {${', '.join(arg_elem(i, arg) for i, arg in enumerate(args))}};
  godot::api->godot_method_bind_ptrcall(mb, o, args, ${'&ret' if ret != 'void' else 'NULL'});

  % if ret != 'void':
  % if is_class_type(ret):
  if (ret) {
    ret = (PyObject *)godot::nativescript_1_1_api->godot_nativescript_get_instance_binding_data(godot::_RegisterState::python_language_index, ret);
    Py_XINCREF(ret);
  }
  return ret;
  % else:
  return ret;
  % endif ## is_call_type
  % endif ## ret != void
  % endif ## has_varargs
}
% endfor
}
#endif
