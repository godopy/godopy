import types

import godot as gd
import gdextension as gde


__all__ = [
    'method',
    'virtual_method',

    'GodotClassBase',
    'Class',
    'EngineClass',
    'EngineObject',
    'Extension',
]


def method(func):
    func._gdmethod = func.__name__
    return func


def virtual_method(func):
    func._gdvirtualmethod = func.__name__
    return func


class GodotClassBase(type):
    """Metaclass for all Godot engine and extension classes"""

    def __new__(cls, name, bases, attrs, **kwargs):
        super_new = super().__new__

        godot_cls = attrs.get('__godot_class__', None)
        if godot_cls is not None:
            # Engine class
            cls._is_extension = False
            gd.print('Setup Engine class %s' % name)
            return super_new(cls, name, bases, attrs)

        # Ensure initialization is only performed for subclasses of Godot classes
        # (excluding Class/EngineClass).
        parents = [b for b in bases if isinstance(b, GodotClassBase)]
        if not parents:
            if name != 'Class' and name != 'EngineClass':
                gd.push_warning("Attempt to create %r Godot class without a valid base" % name)
            # Create Class or EngineClass
            return super_new(cls, name, bases, attrs)

        inherits = kwargs.pop('inherits', None)
        if inherits is None:
            raise TypeError("'inherits' keyword argument is required in Godot class definitions")

        godot_cls = gde.ExtensionClass(name, inherits.__godot_class__)

        module = attrs.pop('__module__')
        new_attrs = {
            '__module__': module,
            '__godot_class__': godot_cls
        }

        cls._is_extension = True

        for attr, value in attrs.items():
            parent_method_info = inherits.get_method_info(attr)
            if attr == '__init__':
                value.__name__ = value.__qualname__ = '__inner_init__'
                # print("Set up __init__(%r) of %s" % (value, name))
                new_attrs['__inner_init__'] = godot_cls.bind_python_method(value)

            # Virtual functions have no hash
            elif parent_method_info is not None and parent_method_info['hash'] is None:
                gd.print('Meta: FOUND VIRTUAL %s.%s %r' % (godot_cls.__inherits__.__name__, attr, parent_method_info))
                new_attrs[attr] = godot_cls.bind_virtual_method(value)

            elif attr.startswith('_') and isinstance(value, types.FunctionType):
                # Warn users about _functions that are not virtuals
                gd.push_warning("No virtual method %r found in the base class %r" % (attr, inherits))
            elif getattr(value, '_gdmethod', False):
                new_attrs[attr] = godot_cls.bind_method(value)
            elif getattr(value, '_gdvirtualmethod', False):
                new_attrs[attr] = godot_cls.add_virtual_method(value)
            elif isinstance(value, types.FunctionType):
                new_attrs[attr] = godot_cls.bind_python_method(value)
            else:
                new_attrs[attr] = value

        return super_new(cls, name, bases, new_attrs, **kwargs)

    def __getattr__(cls, name):
        return getattr(cls.__godot_class__, name)


class Extension(gde.Extension):
    def __getattr__(self, name):
        try:
            mb = gde.MethodBind(self, name)

            self.__dict__[name] = mb

            return mb
        except AttributeError as exc:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, name))

    def __dir__(self):
        return list(set(
            self.__godot_class__.__method_info__.keys() +
            self.__dict__.keys()
        ))


class Class(Extension, metaclass=GodotClassBase):
    def __init__(self):
        godot_cls = self.__class__.__godot_class__
        super().__init__(godot_cls, godot_cls.__inherits__)

class EngineObject(gde.Object):
    def __getattr__(self, name):
        try:
            mb = gde.MethodBind(self, name)

            self.__dict__[name] = mb

            return mb
        except AttributeError as exc:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, name))


    def __dir__(self):
        return self.__godot_class__.__method_info__.keys()


class EngineClass(EngineObject, metaclass=GodotClassBase):
    def __init__(self):
        godot_cls = self.__class__.__godot_class__
        super().__init__(godot_cls)
