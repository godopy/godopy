from typing import Any, List, Mapping, Optional, Sequence, Tuple

import gdextension

import godot
from godot import classdb
from godot.types import String
from godot.classdb import (
    Engine,
    Node as _Node,
    PackedScene, Resource, ResourceLoader, ResourceSaver,
    EditorPlugin
)


_scene_cache = {}


class Node:
    def __init__(self, node_class: godot.GodotClassBase,
                 children: Optional[Sequence['Node']] = None,
                 **properties: Any) -> None:
        self.node_class = node_class
        self._name = node_class.__name__
        self.children = children
        self.properties = properties

    def set_name(self, name: str) -> str:
        self._name = name
        return name

    def get_name(self) -> str:
        return self._name

    name = property(get_name, set_name)


class ExtResource:
    def __init__(self, path: str, hint: str = '', cache_mode=1):
        self._key = hash(self)
        self.path = path
        self.hint = hint
        self.cache_mode = cache_mode

    def set_key(self, key: Any) -> Any:
        self._key = key
        return key

    def get_key(self) -> Any:
        return self._key

    key = property(get_key, set_key)

    def load(self) -> Resource:
        return ResourceLoader.load(self.path, self.hint, self.cache_mode)


class ExtScene(ExtResource):
    def __init__(self, path: str, cache_mode=1):
        super().__init__(path, 'PackedScene', cache_mode)


class SubResource:
    def __init__(self, resource_class: godot.GodotClassBase, **properties: Any):
        self._key = hash(self)
        self.resource_class = resource_class
        self.properties = properties

    def set_key(self, key: Any) -> Any:
        self._key = key
        return key

    def get_key(self) -> Any:
        return self._key

    key = property(get_key, set_key)


class NodeGroup:
    def __init__(self, name: str) -> None:
        self.nodes = []
        self.nodes_by_type = {}
        self.name = name

    def create(self, node_class: godot.GodotClassBase, name: Optional[str] = None,
               children: Optional[Sequence['Node']] = None,
               **properties: Any) -> Node:
        node = Node(node_class, children=children, **properties)
        typed_nodes: list = self.nodes_by_type.setdefault(node_class.__name__, [])
        index = len(typed_nodes)
        node.name = name or node_class.__name__ + index
        self.nodes.append(node)
        typed_nodes.append(node)

        return node


class SceneMeta(type):
    def __new__(metacls, name: str, bases: tuple, attrs: Mapping[str, Any]) -> type:
        class_name = name
        root = attrs.pop('__root_class__', _Node)
        path = attrs.pop('__path__', None)

        if path is None:
            path = 'res://' + String(name).to_snake_case().replace('_scene', '') + '.tscn'

        new_attrs = {}
        node_list = []
        ext_resource_list = []
        sub_resource_list = []

        for name, attr in attrs.items():
            if isinstance(attr, Node):
                attr.name = name
                node_list.append(attr)
            elif isinstance(attr, ExtResource):
                attr.key = name
                ext_resource_list.append(attr)
            # Check for Scene, Scene class is not available yet
            elif hasattr(attr.__class__, '__root_class__'):
                node_list.append(attr)
            elif isinstance(attr, ExtScene):
                node_list.append(attr)
            elif isinstance(attr, SubResource):
                attr.key = name
                sub_resource_list.append(attr)
            if isinstance(attr, type):
                group = NodeGroup(attr.__name__)

                for name, attr in attr.__dict__.items():
                    if name in ('__dict__', '__weakref__'):
                        continue
                    if isinstance(attr, Node):
                        group.create(attr.node_class, name, attr.children, **attr.properties)
                    elif isinstance(attr, ExtResource):
                        attr.key = name
                        ext_resource_list.append(attr)
                    elif isinstance(attr, SubResource):
                        attr.key = name
                        sub_resource_list.append(attr)
                node_list.append(group)
            else:
                new_attrs[name] = attr

        new_attrs['__root_class__'] = root
        new_attrs['__path__'] = path
        new_attrs['__node_list__'] = node_list
        new_attrs['__ext_resource_list__'] = ext_resource_list
        new_attrs['__sub_resource_list__'] = sub_resource_list

        cls = super().__new__(metacls, class_name, bases, new_attrs)

        _scene_cache[path] = cls

        return cls


_deferred_initializers = []
_deferred_node_cleanup = []

class Scene(metaclass=SceneMeta):
    __root_class__ = _Node

    def __init__(self, name=None, **properties):
        cls = self.__class__

        if name is None:
            if hasattr(cls, 'name'):
                name = cls.name
            else:
                name = cls.__name__
                if name.endswith('Scene'):
                    name = name[:-5] or self.__root_class__.__name__

        self.name = name

        # Cannot initialize too early, must wait until the Engine is fully loaded
        _deferred_initializers.append((self._init_scene, (name, properties)))


    def _init_scene(self, name, **properties):
        cls = self.__class__

        if ResourceLoader.exists(self.__path__, 'PackedScene'):
            self.packed = ResourceLoader.load(self.__path__, 'PackedScene', 1)
            self.root = self.packed.instantiate()
            if Engine.is_editor_hint():
                # Sync scenes only when the Editor is active
                self.root.set_name(name)

                _meta = properties.pop('_meta', {})
                for name, value in properties.items():
                    self.root.set(name, value)
                for name, value in _meta.items():
                    self.root.set_meta(name, value)
            else:
                self.name = self.root.get_name()

        elif Engine.is_editor_hint():
            # Create scenes only when the Editor is active
            self.packed = PackedScene()
            self.root = self.__root_class__(name=name, **properties)
            self.packed.reference()
        else:
            self.packed = None
            self.root = None

        if Engine.is_editor_hint():
            # Sync scenes only when the Editor is active

            self._res = {}

            _added_nodes = []

            # Resource loading function for the 'load' method
            def load_resource(key: Any, path_or_class: str | godot.GodotClassBase, *args, **properties) -> Resource:
                if isinstance(path_or_class, godot.GodotClassBase):
                    resource_class = path_or_class
                    props = self.get_properties(properties)
                    resource = resource_class(**props)
                    resource.reference()
                else:
                    path = path_or_class
                    if len(args) == 2:
                        hint, cache_mode = args
                    elif len(args) == 1:
                        hint = args[0]
                        cache_mode = properties.pop('cache_mode', 1)
                    elif args:
                        raise TypeError(f"load_resource got an unexpected positional argument f{args[0]}")
                    else:
                        hint = properties.pop('hint', '')
                        cache_mode = properties.pop('cache_mode', 1)
                    if properties:
                        raise TypeError(f"load_resource got an unexpected keyword argument f{list(properties)[0]}")
                    resource = ResourceLoader.load(path, hint, cache_mode)

                self._res[key] = resource
                return resource

            # load resources defined in the 'load' method
            if hasattr(self, 'load') and callable(self.load):
                self.load(load_resource)

            # load external resources defined in the class body
            for ext_resource in cls.__ext_resource_list__:
                resource = ext_resource.load()
                self._res[ext_resource.key] = resource

            # load sub resources defined in the class body
            for sub_resource in cls.__sub_resource_list__:
                props = self.get_properties(sub_resource.properties)
                resource = sub_resource.resource_class(**props)
                resource.reference()
                self._res[sub_resource.key] = resource

            # add nodes defined in the 'create' function
            if hasattr(self, 'create') and callable(self.load):
                self.create(self._res.__getitem__)

            # add nodes defined in the class body
            for node in cls.__node_list__:
                if isinstance(node, Scene):
                    self._add_scene(node)
                elif isinstance(node, NodeGroup):
                    group_node = Node(_Node, children=node.nodes)
                    group_node.name = node.name
                    child, granchildren = self._add(group_node)
                    _added_nodes.append(child)
                    _added_nodes.extend(granchildren)
                else:
                    child, granchildren = self._add(node)
                    _added_nodes.append(child)
                    _added_nodes.extend(granchildren)

            result = self.packed.pack(self.root)
            if result != godot.Error.OK:
                msg = f"An error ({godot.Error(result)!r}) occurred while packing the scene ({self.root!r})."

                raise RuntimeError(msg)

            result = ResourceSaver.save(self.packed, self.__path__, 0)
            if result != godot.Error.OK:
                msg = f"An error ({godot.Error(result)!r}) occurred while saving the scene to disk."

                raise RuntimeError(msg)

            _deferred_node_cleanup.extend(_added_nodes)
            del _added_nodes

        self._res = None
        self.packed = None
        self.root = None


    def get_properties(self, properties, existing_node=None):
        props = {}
        for name, prop in properties.items():
            if isinstance(prop, ExtResource):
                props[name] = self._res.get(prop.key, prop.load())
            elif isinstance(prop, SubResource):
                if existing_node:
                    available_names = {p['name'] for p in existing_node.get_property_list()}
                    if name not in available_names:
                        raise TypeError(f'Trying to access non-existent property {name!r}')
                    if name == 'material':
                        # XXX: get('material') can crash and it is probably a typo
                        gdextension.print_warning("Trying to set the 'material' property, did you mean 'physics_material_override'?")
                        existing = existing_node.get_material()
                    else:
                        existing = existing_node.get(name)
                else:
                    existing = None
                if existing:
                    res = existing
                    res_props = self.get_properties(prop.properties)
                    for key, value in res_props.items():
                        res.set(key, value)
                else:
                    res = self._res.get(prop.key, prop.resource_class(**self.get_properties(prop.properties)))
                props[name] = res
                res.reference()
            elif isinstance(prop, dict):
                props[name] = self.get_properties(prop)
            elif isinstance(prop, list):
                list_value = []
                for item in prop:
                    if isinstance(item, ExtResource):
                        list_value.append(self._res.get(item.key, item.load()))
                    elif isinstance(item, SubResource):
                        res = self._res.get(
                            item.key,
                            item.resource_class(**self.get_properties(item.properties))
                        )
                        res.reference()
                        list_value.append(res)

                    elif isinstance(item, dict):
                        list_value.append(self.get_properties(item))
                    else:
                        list_value.append(prop)
                props[name] = list_value
            else:
                props[name] = prop

        return props


    @staticmethod
    def from_path(path):
        CachedClass = _scene_cache.get(path, None)
        if CachedClass is not None:
            return CachedClass()

        return ExtScene(path)


    def get_path(self):
        return self.__path__


    def __repr__(self):
        return f"<Scene object {self.name!r} of type {self.__root_class__}>"


    def __getattr__(self, attr):
        if attr.startswith('add_') and self.root is not None:
            class_name = attr[4:]  # attr[4:].title()
            cls = getattr(classdb, class_name)

            def add_child(name=None, **attrs):
                node = Node(cls, attrs)
                node.name = name or class_name
                self._add(node)

            return add_child

        raise AttributeError(f"{self.__class__!r} object has no attribute {attr!r}")

    def _add(self, node: Node, to_parent=None, existing_node=None) -> Tuple[_Node, List[_Node]]:
        owner = self.root
        root = owner if to_parent is None else to_parent

        if root is None:
            raise RuntimeError("Cannot add nodes before or after Scene's initialization")

        existing = existing_node or root.find_child(node.name, False, True)
        props = self.get_properties(node.properties, existing)

        if existing:
            child = existing
            _meta = props.pop('_meta', {})
            for key, value in props.items():
                child.set(key, value)
            for key, value in _meta.items():
                child.set_meta(key, value)
        else:
            child = node.node_class(name=node.name, **props)
            root.add_child(child, False, 0)
            child.set_owner(owner)

        _added_nodes = []

        if node.children is not None:
            existing = child.get_children(True)
            existing_dict = {node.get_name(): node for node in existing}
            # print('found existing', existing_dict)
            for childnode in node.children:
                grandchild, grandgrandchildren = self._add(
                    childnode,
                    to_parent=child,
                    existing_node=existing_dict.get(childnode.name, None)
                )
                _added_nodes.append(grandchild)
                _added_nodes.extend(grandgrandchildren)

        return child, _added_nodes

    def _add_scene(self, scene: 'Scene'):
        root = self.root

        existing = root.find_child(scene.name, False, True)

        if not existing:
            packed = ResourceLoader.load(scene.get_path(), 'PackedScene', 1)
            child = packed.instantiate()
            root.add_child(child, False, 0)
            child.set_owner(root)

class PythonScenePlugin(godot.Class, inherits=EditorPlugin):
    def _enter_tree(self):
        global _deferred_initializers

        for func, (name, properties) in _deferred_initializers:
            func(name, **properties)

        del _deferred_initializers

    def _exit_tree(self):
        global _scene_cache, _deferred_node_cleanup
        del _scene_cache

        _deferred_node_cleanup.reverse()

        for node in _deferred_node_cleanup:
            node.destroy()

        del _deferred_node_cleanup
