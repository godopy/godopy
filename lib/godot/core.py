import types

import godot as gd
import gdextension as gde


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

    def __new__(cls, name, bases, attrs, **kwargs):
        super_new = super().__new__

        godot_cls = attrs.pop('__godot_class__', None)
        if godot_cls is not None:
            # Engine class
            module = attrs.pop('__module__', 'godot.classdb')

            if module == 'godot.classdb' and gde._has_singleton(name):
                module = 'godot.singletons'

            new_attrs = {
                '__module__': module,
                '_is_extension': False,
                '__godot_class__': godot_cls
            }

            attrs.update(new_attrs)

            # gd.print('Setup Engine class %s' % name)
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
            '_is_extension': True,
            '__godot_class__': godot_cls
        }

        for attr, value in attrs.items():
            parent_method_info = inherits.get_method_info(attr)
            if attr == '__init__':
                value.__name__ = value.__qualname__ = '__inner_init__'
                # print("Set up __init__(%r) of %s" % (value, name))
                new_attrs['__inner_init__'] = godot_cls.bind_python_method(value)

            # Virtual functions have no hash
            elif parent_method_info is not None and parent_method_info['hash'] is None:
                # gd.print('Meta: FOUND VIRTUAL %s.%s %r' % (godot_cls.__inherits__.__name__, attr, parent_method_info))
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

        new_cls = super_new(cls, name, bases, new_attrs, **kwargs)
        _ext_class_cache[godot_cls] = new_cls

        return new_cls

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


# semi-public, not in __all__, for internal use
def _class_from_godot_class(godot_cls):
    return _ext_class_cache[godot_cls]

class Class(Extension, metaclass=GodotClassBase):
    def __init__(self, **kwargs):
        has_kwargs = list(kwargs.keys())

        godot_cls = kwargs.pop('__godot_class__', self.__class__.__godot_class__)
        _notify = kwargs.pop('_notify', True)
        _from_callback = kwargs.pop('_from_callback', False)
        _internal_check = kwargs.pop('_internal_check', '')

        # kwargs are for internal use only, add some protection
        msg = "%s.__init__() got an unexpected keyword argument %r"
        if has_kwargs and _internal_check != hex(id(godot_cls)):
            raise TypeError(msg % (self.__godot_class__, has_kwargs.pop()))

        if kwargs:
            raise TypeError(msg % (self.__godot_class__, list(kwargs.keys()).pop()))

        super().__init__(godot_cls, godot_cls.__inherits__, _notify, _from_callback)


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
    def __init__(self, from_ptr=0):
        godot_cls = self.__class__.__godot_class__
        super().__init__(godot_cls, from_ptr=from_ptr)
