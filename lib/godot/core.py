"""
Provides GodotClassBase metaclass and base classes for all Godot classes and objects.
"""
import types
from typing import Any, Callable, List, Mapping

import gdextension


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
                try:
                    return getattr(godot_cls, attr)
                except AttributeError:
                    raise AttributeError(f"'{module}.{name}' has no attribute {attr!r}")

            def add_property(self, info: gdextension.PropertyInfo, *args) -> None:
                all_attrs[info.name] = [info, *args]

            def bind_method(self, func: Callable) -> Callable:
                func._gdmethod = func.__name__
                return func

            def bind_virtual_method(self, func: Callable) -> Callable:
                func._gdvirtualmethod = func.__name__
                return func

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
            elif isinstance(value, list) and isinstance(value[0], gdextension.PropertyInfo):
                godot_cls.add_property(*value)
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
        has_kwargs = list(kwargs.keys())

        godot_cls = kwargs.pop('__godot_class__', self.__class__.__godot_class__)
        from_callback = kwargs.pop('from_callback', False)
        _internal_check = kwargs.pop('_internal_check', '')

        # kwargs are for internal use only, add some protection
        msg = "%s.__init__() got an unexpected keyword argument %r"
        if has_kwargs and _internal_check != hex(id(godot_cls)):
            raise TypeError(msg % (self.__godot_class__, has_kwargs.pop()))

        if kwargs:
            raise TypeError(msg % (self.__godot_class__, list(kwargs.keys()).pop()))

        super().__init__(godot_cls, from_callback=from_callback)


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
    def __init__(self, from_ptr : int = 0) -> None:
        godot_cls = self.__class__.__godot_class__
        super().__init__(godot_cls, from_ptr=from_ptr)
