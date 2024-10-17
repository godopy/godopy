#include <Python.h>
#include <structmember.h>
#include <structseq.h>

PyTypeObject Vector2_Type = {0, 0, 0, 0, 0, 0};
PyTypeObject Vector2i_Type = {0, 0, 0, 0, 0, 0};
PyTypeObject Size2_Type = {0, 0, 0, 0, 0, 0};
PyTypeObject Rect2_Type = {0, 0, 0, 0, 0, 0};
PyTypeObject Rect2i_Type = {0, 0, 0, 0, 0, 0};
PyTypeObject Vector3_Type = {0, 0, 0, 0, 0, 0};
PyTypeObject Vector3i_Type = {0, 0, 0, 0, 0, 0};
PyTypeObject Vector4_Type = {0, 0, 0, 0, 0, 0};
PyTypeObject Vector4i_Type = {0, 0, 0, 0, 0, 0};
PyTypeObject Plane_Type = {0, 0, 0, 0, 0, 0};
PyTypeObject Quaternion_Type = {0, 0, 0, 0, 0, 0};
PyTypeObject AABB_Type = {0, 0, 0, 0, 0, 0};
PyTypeObject Transform3D_Type = {0, 0, 0, 0, 0, 0};
PyTypeObject Color_Type = {0, 0, 0, 0, 0, 0};

static PyStructSequence_Field vector2_fields[] = {
    {"x", "x coordinate"},
    {"y", "y coordinate"},
    {NULL}
};

static PyStructSequence_Field size2_fields[] = {
    {"width", "Width"},
    {"height", "Height"},
    {NULL}
};

static PyStructSequence_Field rect2_fields[] = {
    {"position", "Position"},
    {"size", "Size"},
    {NULL}
};

static PyStructSequence_Field vector3_fields[] = {
    {"x", "x coordinate"},
    {"y", "y coordinate"},
    {"z", "z coordinate"},
    {NULL}
};

static PyStructSequence_Field vector4_fields[] = {
    {"x", "x coordinate"},
    {"y", "y coordinate"},
    {"z", "z coordinate"},
    {"w", "w coordinate"},
    {NULL}
};

static PyStructSequence_Field plane_fields[] = {
    {"normal", "Normal vector"},
    {"d", "Distance from the origin to the plane"},
    {NULL}
};

static PyStructSequence_Field transform3d_fields[] = {
    {"basis", "Basis"},
    {"origin", "Origin"},
    {NULL}
};

static PyStructSequence_Field color_fields[] = {
    {"r", "Red component"},
    {"g", "Green component"},
    {"b", "Blue component"},
    {"a", "Alpha (aka opacity) component"},
    {NULL}
};

static PyStructSequence_Desc vector2_desc = {
    "Vector2",
    NULL,
    vector2_fields,
    2
};

static PyStructSequence_Desc vector2i_desc = {
    "Vector2i",
    NULL,
    vector2_fields,
    2
};

static PyStructSequence_Desc size2_desc = {
    "Size2",
    NULL,
    size2_fields,
    2
};

static PyStructSequence_Desc rect2_desc = {
    "Rect2",
    NULL,
    rect2_fields,
    2
};

static PyStructSequence_Desc rect2i_desc = {
    "Rect2i",
    NULL,
    rect2_fields,
    2
};

static PyStructSequence_Desc vector3_desc = {
    "Vector3",
    NULL,
    vector3_fields,
    3
};

static PyStructSequence_Desc vector3i_desc = {
    "Vector3i",
    NULL,
    vector3_fields,
    3
};

static PyStructSequence_Desc vector4_desc = {
    "Vector4",
    NULL,
    vector4_fields,
    4
};

static PyStructSequence_Desc vector4i_desc = {
    "Vector4i",
    NULL,
    vector4_fields,
    4
};

static PyStructSequence_Desc plane_desc = {
    "Plane",
    NULL,
    plane_fields,
    2
};

static PyStructSequence_Desc quaternion_desc = {
    "Quaternion",
    NULL,
    vector4_fields,
    4
};

static PyStructSequence_Desc aabb_desc = {
    "AABB",
    NULL,
    rect2_fields,
    2
};

static PyStructSequence_Desc transform3d_desc = {
    "Transform3D",
    NULL,
    transform3d_fields,
    2
};

static PyStructSequence_Desc color_desc = {
    "Color",
    NULL,
    color_fields,
    4
};

static struct PyModuleDef py_godot_types_module = {
    PyModuleDef_HEAD_INIT,
    "_godot_type_bases",
    "Collection of named tuples (PyStructSequences) to represent various Godot engine types",
    -1,
    NULL
};

extern "C" PyMODINIT_FUNC PyInit__godot_type_tuples(void) {
    PyObject *mod = PyModule_Create(&py_godot_types_module);

    if (mod == nullptr) {
        return nullptr;
    }

    if (Vector2_Type.tp_name == 0) { PyStructSequence_InitType(&Vector2_Type, &vector2_desc); }
    if (Vector2i_Type.tp_name == 0) { PyStructSequence_InitType(&Vector2i_Type, &vector2i_desc); }
    if (Size2_Type.tp_name == 0) { PyStructSequence_InitType(&Size2_Type, &size2_desc); }
    if (Rect2_Type.tp_name == 0) { PyStructSequence_InitType(&Rect2_Type, &rect2_desc); }
    if (Rect2i_Type.tp_name == 0) { PyStructSequence_InitType(&Rect2i_Type, &rect2i_desc); }
    if (Vector3_Type.tp_name == 0) { PyStructSequence_InitType(&Vector3_Type, &vector3_desc); }
    if (Vector3i_Type.tp_name == 0) { PyStructSequence_InitType(&Vector3i_Type, &vector3i_desc); }
    if (Vector4_Type.tp_name == 0) { PyStructSequence_InitType(&Vector4_Type, &vector4_desc); }
    if (Vector4i_Type.tp_name == 0) { PyStructSequence_InitType(&Vector4i_Type, &vector4i_desc); }
    if (Plane_Type.tp_name == 0) { PyStructSequence_InitType(&Plane_Type, &plane_desc); }
    if (Quaternion_Type.tp_name == 0) { PyStructSequence_InitType(&Quaternion_Type, &quaternion_desc); }
    if (AABB_Type.tp_name == 0) { PyStructSequence_InitType(&AABB_Type, &aabb_desc); }
    if (Transform3D_Type.tp_name == 0) { PyStructSequence_InitType(&Transform3D_Type, &transform3d_desc); }
    if (Color_Type.tp_name == 0) { PyStructSequence_InitType(&Color_Type, &color_desc); }
        
    Py_INCREF((PyObject *) &Vector2_Type);
    PyModule_AddObject(mod, "Vector2", (PyObject *) &Vector2_Type);

    Py_INCREF((PyObject *) &Vector2i_Type);
    PyModule_AddObject(mod, "Vector2i", (PyObject *) &Vector2i_Type);

    Py_INCREF((PyObject *) &Size2_Type);
    PyModule_AddObject(mod, "Size2", (PyObject *) &Size2_Type);

    Py_INCREF((PyObject *) &Rect2_Type);
    PyModule_AddObject(mod, "Rect2", (PyObject *) &Rect2_Type);

    Py_INCREF((PyObject *) &Rect2i_Type);
    PyModule_AddObject(mod, "Rect2i", (PyObject *) &Rect2i_Type);

    Py_INCREF((PyObject *) &Vector3_Type);
    PyModule_AddObject(mod, "Vector3", (PyObject *) &Vector3_Type);

    Py_INCREF((PyObject *) &Vector3i_Type);
    PyModule_AddObject(mod, "Vector3i", (PyObject *) &Vector3i_Type);

    Py_INCREF((PyObject *) &Vector4_Type);
    PyModule_AddObject(mod, "Vector4", (PyObject *) &Vector4_Type);

    Py_INCREF((PyObject *) &Vector4i_Type);
    PyModule_AddObject(mod, "Vector4i", (PyObject *) &Vector4i_Type);

    Py_INCREF((PyObject *) &Plane_Type);
    PyModule_AddObject(mod, "Plane", (PyObject *) &Plane_Type);

    Py_INCREF((PyObject *) &Quaternion_Type);
    PyModule_AddObject(mod, "Quaternion", (PyObject *) &Quaternion_Type);

    Py_INCREF((PyObject *) &AABB_Type);
    PyModule_AddObject(mod, "AABB", (PyObject *) &AABB_Type);

    Py_INCREF((PyObject *) &Transform3D_Type);
    PyModule_AddObject(mod, "Transform3D", (PyObject *) &Transform3D_Type);

    Py_INCREF((PyObject *) &Color_Type);
    PyModule_AddObject(mod, "Color", (PyObject *) &Color_Type);

    return mod;
}
