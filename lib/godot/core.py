"""
Provides GodotClassBase metaclass and base classes for all Godot classes and objects.
"""
import enum
import types
from typing import Any, Callable, List, Mapping

import gdextension
from gdextension import PropertyInfo as _PropertyInfo

__all__ = [
    'GodotClassBase',
    'Class',
    'GDREGISTER_CLASS',
    'GDCLASS',
    'EngineClass',
    'EngineObject',
    'Extension',
]


_ext_class_cache = {}

class GodotClassBase(type):
    """Metaclass for all Godot engine and extension classes"""

    def __new__(cls, name: str, bases: tuple, attrs: Mapping[str, Any], **kwargs: Any) -> type:
        super_new = super().__new__

        godot_cls = attrs.pop('__godot_class__', None)
        if godot_cls is not None:
            if kwargs:
                raise TypeError(f"{cls.__name__!r} got an unexpected keyword argument {list(kwargs).pop()!r}")

            # Engine class
            module = attrs.pop('__module__', 'godot.classdb')

            new_attrs = {
                '__module__': module,
                '_is_extension': False,
                '__godot_class__': godot_cls
            }

            attrs.update(new_attrs)

            return super_new(cls, name, bases, attrs)

        # Ensure initialization is only performed for subclasses of Godot classes
        # (excluding Class/EngineClass).
        parents = [b for b in bases if isinstance(b, GodotClassBase)]
        if not parents:
            if name != 'Class' and name != 'EngineClass':
                gdextension.print_warning(f"Attempt to create {name!r} Godot class without a valid base")

            if kwargs:
                raise TypeError(f"{cls.__name__!r} got an unexpected keyword argument {list(kwargs).pop()!r}")

            # Create Class or EngineClass
            return super_new(cls, name, bases, attrs)

        inherits = kwargs.pop('inherits', None)
        if inherits is None:
            raise TypeError("'inherits' keyword argument is required in Godot class definitions")

        # If set, reverses the meaning of single leading underscores
        no_virtual_underscore = kwargs.pop('no_virtual_underscore', False)
        mixins = kwargs.pop('mixins', [])

        if kwargs:
            raise TypeError(f"{cls.__name__!r} got an unexpected keyword argument {list(kwargs).pop()!r}")

        godot_cls = gdextension.ExtensionClass(name, inherits.__godot_class__)

        module = attrs.pop('__module__')
        new_attrs = {
            '__module__': module,
            '_is_extension': True,
            '__godot_class__': godot_cls
        }

        all_attrs = {}
        for mixin in mixins:
            all_attrs.update(mixin.__dict__)

        all_attrs.update(attrs)

        is_virtual = lambda info: info['is_virtual']

        _bind_methods_func = all_attrs.pop('_bind_methods', None)
        if hasattr(_bind_methods_func, '__func__'):
            # Unbind if necessary
            _bind_methods_func = _bind_methods_func.__func__

        class BindMethodsClassPlaceholder:
            def __getattr__(sef, attr):
                if attr in all_attrs:
                    return all_attrs[attr]
                value = getattr(godot_cls, attr, None)
                if value is None:
                    raise AttributeError(f"'{module}.{name}' has no attribute {attr!r}")
                return value

            def bind_method(self, func: Callable) -> Callable:
                func._gdmethod = func.__name__
                return func

            def bind_virtual_method(self, func: Callable) -> Callable:
                func._gdvirtualmethod = func.__name__
                return func

            def bind_int_enum(self, enum_obj: enum.IntEnum) -> enum.IntEnum:
                all_attrs[f"intenum${enum_obj.__name__}"] = enum_obj
                return enum_obj

            def bind_bitfield(self, enum_obj: enum.IntEnum) -> enum.IntEnum:
                all_attrs[f"bitfield${enum_obj.__name__}"] = enum_obj
                return enum_obj

            def add_property(self, info: _PropertyInfo, *args) -> _PropertyInfo:
                all_attrs[f"prop${info.name}"] = [info, *args]
                return info

            def add_property_i(self, info: _PropertyInfo, *args) -> _PropertyInfo:
                all_attrs[f"idxprop${info.name}"] = [info, *args]
                return info

            def add_group(self, group_name: str, prefix: str = '') -> str:
                all_attrs[f"group${group_name}"] = [group_name, prefix]
                return group_name

            def add_subgroup(self, subgroup_name: str, prefix: str = '') -> str:
                all_attrs[f"group${subgroup_name}"] = [subgroup_name, prefix]
                return subgroup_name

            def add_signal(self, signal_name: str, *arguments: _PropertyInfo) -> str:
                all_attrs[f"signal${signal_name}"] = [signal_name, arguments]
                return signal_name


        if _bind_methods_func is not None:
            _bind_methods_func(BindMethodsClassPlaceholder())

        for attr, value in all_attrs.items():
            inner_attr_name = attr
            parent_method_info = inherits.get_method_info(attr)

            if no_virtual_underscore:
                underscored_attr = f'_{attr}'
                old_info = parent_method_info
                parent_method_info = inherits.get_method_info(underscored_attr)
                if parent_method_info is not None and is_virtual(parent_method_info):
                    inner_attr_name = underscored_attr
                else:
                    # Allow underscores even in 'no_virtual_underscore' mode
                    parent_method_info = old_info

            if attr == '__init__':
                new_attrs['__inner_init__'] = godot_cls.bind_python_method(value, '__inner_init__')

            elif parent_method_info is not None and is_virtual(parent_method_info):
                # godot.print(f'[Virtual] {godot_cls.__inherits__.__name__}.{inner_attr_name} {parent_method_info!r}')
                if inner_attr_name != attr:
                    value._alias_of = inner_attr_name
                new_attrs[inner_attr_name] = godot_cls.bind_virtual_method(value, inner_attr_name)

            elif not no_virtual_underscore and attr.startswith('_') and isinstance(value, types.FunctionType):
                # Warn users about _functions that are not virtuals
                gdextension.print_warning(f"No virtual method {attr!r} found in the base class {inherits!r}")
            elif no_virtual_underscore and not attr.startswith('_') and isinstance(value, types.FunctionType):
                # Warn users about functions that are not virtuals in 'no_virtual_underscore' mode
                gdextension.print_warning(f"No virtual method '_{attr}' found in the base class {inherits!r}")
            elif getattr(value, '_gdmethod', False):
                new_attrs[attr] = godot_cls.bind_method(value)
            elif getattr(value, '_gdvirtualmethod', False):
                new_attrs[attr] = godot_cls.add_virtual_method(value)
            elif attr.startswith('intenum$'):
                godot_cls.bind_int_enum(value)
            elif attr.startswith('bitfield$'):
                godot_cls.bind_bitfield(value)
            elif attr.startswith('prop$'):
                godot_cls.add_property(*value)
            elif attr.startswith('idxprop$'):
                godot_cls.add_property_i(*value)
            elif attr.startswith('group$'):
                godot_cls.add_group(*value)
            elif attr.startswith('subgroup$'):
                godot_cls.add_subgroup(*value)
            elif attr.startswith('signal$'):
                godot_cls.add_signal(*value)
            elif isinstance(value, types.FunctionType):
                new_attrs[attr] = godot_cls.bind_python_method(value)
            else:
                new_attrs[attr] = value

        new_cls = super_new(cls, name, bases, new_attrs, **kwargs)
        _ext_class_cache[godot_cls] = new_cls

        return new_cls

    def __getattr__(cls, name):
        return getattr(cls.__godot_class__, name)


class Extension(gdextension.Extension):
    """
    Base class for all custom GDExtension objects.
    """
    def __getattr__(self, name) -> Any:
        try:
            mb = gdextension.MethodBind(self, name)

            self.__dict__[name] = mb

            return mb
        except AttributeError as exc:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, name))

    def __dir__(self) -> List[str]:
        return list(set(
            list(self.__godot_class__.__method_info__.keys()) +
            [k for k in self.__dict__.keys() if not k.stratswith('_')]
        ))

    def cast_to(self, class_name: str) -> None:
        from godot import classdb

        super().cast_to(class_name)
        self._switch_class(getattr(classdb, class_name))

class Class(Extension, metaclass=GodotClassBase):
    """
    Base class for all custom GDExtension classes.
    """
    def __init__(self, **kwargs) -> None:
        internal_kwarg = kwargs.get('__godot_class__', kwargs.get('from_callback', None))

        godot_cls = kwargs.pop('__godot_class__', self.__class__.__godot_class__)
        from_callback = kwargs.pop('from_callback', False)
        _internal_check = kwargs.pop('_internal_check', '')

        # Add some protection for internal keyword arguments
        msg = "%s.__init__() got an unexpected keyword argument %r"
        if internal_kwarg is not None and _internal_check != hex(id(godot_cls)):
            raise TypeError(msg % (self.__godot_class__, internal_kwarg))

        super().__init__(godot_cls, from_callback=from_callback)

        if kwargs:
            get_property_list = gdextension.MethodBind(self, 'get_property_list')
            props = {p['name'] for p in get_property_list()}
            props |= {p['name'] for p in self.get_property_list()}

            _meta = kwargs.pop('_meta', {})

            for key, value in kwargs.items():
                if key not in props:
                    msg = f"{self.__class__.__name__}.__init__() got an unexpected keyword argument {key!r}"
                    raise TypeError(msg)

                if value is not None:
                    # print('set', self.__class__, key, value)
                    self.set(key, value)

            for key, value in _meta.items():
                self.set_meta(key, value)


def GDREGISTER_CLASS(cls: GodotClassBase) -> None:
    return cls.register()


def GDCLASS(cls: GodotClassBase) -> GodotClassBase:
    if not issubclass(cls, EngineClass):
        raise TypeError("Expected 'EngineClass' as an argument of 'GDCLASS'")

    def decorator(decorated_cls: type):
        slots =  getattr(decorated_cls, '__slots__', [])

        new_cls_dict = {
            k: getattr(decorated_cls, k)
            for k in decorated_cls.__dict__
            if k not in ('__dict__', '__weakref__') and k not in slots
        }

        new_cls = GodotClassBase(
            decorated_cls.__name__,
            (Class,),
            new_cls_dict,
            inherits=cls
        )

        return new_cls

    return decorator


class EngineObject(gdextension.Object):
    """
    Base class for all Engine objects.
    """
    def __getattr__(self, name) -> Any:
        try:
            mb = gdextension.MethodBind(self, name)

            self.__dict__[name] = mb

            return mb
        except AttributeError as exc:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, name))


    def __dir__(self) -> List[str]:
        return self.__godot_class__.__method_info__.keys()


    def cast_to(self, class_name: str) -> None:
        from godot import classdb

        super().cast_to(class_name)

        self._switch_class(getattr(classdb, class_name))


class EngineClass(EngineObject, metaclass=GodotClassBase):
    """
    Base class for all Engine classes.
    """
    def __init__(self, from_ptr : int = 0, **kwargs) -> None:
        godot_cls = self.__class__.__godot_class__

        super().__init__(godot_cls, from_ptr=from_ptr)

        if kwargs:
            props = {p['name'] for p in self.get_property_list()}

            _meta = kwargs.pop('_meta', {})
            for key, value in kwargs.items():
                if key not in props:
                    msg = f"{self.__class__.__name__}.__init__() got an unexpected keyword argument {key!r}"
                    raise TypeError(msg)
                if value is not None:
                    # print('set', self.__class__, key, value)
                    self.set(key, value)
            for key, value in _meta.items():
                self.set_meta(key, value)
