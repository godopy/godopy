#include <Python.h>
#include <structmember.h>
#include <structseq.h>

static PyTypeObject Vector2_Type = {0, 0, 0, 0, 0, 0};
static PyTypeObject Vector3_Type = {0, 0, 0, 0, 0, 0};
static PyTypeObject Vector4_Type = {0, 0, 0, 0, 0, 0};
static PyTypeObject Color_Type = {0, 0, 0, 0, 0, 0};

static PyStructSequence_Field vector2_fields[] = {
    {"x", "x coordinate"},
    {"y", "y coordinate"},
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

static PyStructSequence_Desc vector3_desc = {
    "Vector3",
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

static PyStructSequence_Desc color_desc = {
    "Color",
    NULL,
    color_fields,
    4
};

static struct PyModuleDef py_vector_types_module =
{
    PyModuleDef_HEAD_INIT,
    "_vector_types", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    NULL
};

PyMODINIT_FUNC
initfoo(void)
{
    PyObject *mod = PyModule_Create(&py_vector_types_module);

    if (mod == nullptr) {
        return nullptr;
    }

    if (Vector2_Type.tp_name == 0) {
        PyStructSequence_InitType(&Vector2_Type, &vector2_desc);
    }
    if (Vector3_Type.tp_name == 0) {
        PyStructSequence_InitType(&Vector3_Type, &vector3_desc);
    }
    if (Vector4_Type.tp_name == 0) {
        PyStructSequence_InitType(&Vector4_Type, &vector4_desc);
    }
    if (Color_Type.tp_name == 0) {
        PyStructSequence_InitType(&Color_Type, &color_desc);
    }
        
    // Py_INCREF((PyObject *) &Vector2_Type);
    // PyModule_AddObject(mod, "Vector2", (PyObject *) &Vector2_Type);

    return mod;
}
