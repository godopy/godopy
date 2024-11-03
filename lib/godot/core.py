"""
Provides GodotClassBase metaclass and base classes for all Godot classes and objects.
"""
import types
from typing import Any, List, Mapping, Dict

import godot
import gdextension


__all__ = [
    'GodotClassBase',
    'Class',
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
            # Engine class
            module = attrs.pop('__module__', 'godot.classdb')

            if module == 'godot.classdb' and gdextension.has_singleton(name):
                module = 'godot.singletons'

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
                godot.push_warning("Attempt to create %r Godot class without a valid base" % name)
            # Create Class or EngineClass
            return super_new(cls, name, bases, attrs)

        inherits = kwargs.pop('inherits', None)
        if inherits is None:
            raise TypeError("'inherits' keyword argument is required in Godot class definitions")

        godot_cls = gdextension.ExtensionClass(name, inherits.__godot_class__)

        module = attrs.pop('__module__')
        new_attrs = {
            '__module__': module,
            '_is_extension': True,
            '__godot_class__': godot_cls
        }

        for attr, value in attrs.items():
            parent_method_info = inherits.get_method_info(attr)
            if attr == '__init__':
                value.__name__ = value.__qualname__ = '__inner_init__'
                new_attrs['__inner_init__'] = godot_cls.bind_python_method(value)

            # Virtual functions have no hash
            elif parent_method_info is not None and parent_method_info['hash'] is None:
                # gd.print('Meta: FOUND VIRTUAL %s.%s %r' % (godot_cls.__inherits__.__name__, attr, parent_method_info))
                new_attrs[attr] = godot_cls.bind_virtual_method(value)

            elif attr.startswith('_') and isinstance(value, types.FunctionType):
                # Warn users about _functions that are not virtuals
                godot.push_warning("No virtual method %r found in the base class %r" % (attr, inherits))
            elif getattr(value, '_gdmethod', False):
                new_attrs[attr] = godot_cls.bind_method(value)
            elif getattr(value, '_gdvirtualmethod', False):
                new_attrs[attr] = godot_cls.add_virtual_method(value)
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
