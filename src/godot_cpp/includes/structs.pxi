cdef extern from "godot_cpp/classes/audio_frame.hpp" namespace "godot" nogil:
    cdef struct AudioFrame:
        float left
        float right


cdef extern from "godot_cpp/classes/text_server.hpp" namespace "godot" nogil:
    cdef cppclass TextServer:
        enum Direction:
            DIRECTION_AUTO = 0
            DIRECTION_LTR = 1
            DIRECTION_RTL = 2
            DIRECTION_INHERITED = 3


cdef extern from "godot_cpp/classes/caret_info.hpp" namespace "godot" nogil:
    cdef struct CaretInfo:
        Rect2 leading_caret
        Rect2 trailing_caret
        TextServer.Direction leading_direction
        TextServer.Direction trailing_direction


cdef extern from "godot_cpp/classes/glyph.hpp" namespace "godot" nogil:
    cdef struct Glyph:
        int start  #  # = -1
        int end  # = -1
        uint8_t count  # = 0
        uint8_t repeat  # = 1
        uint16_t flags  # = 0
        float x_off  # = 0.
        float y_off  # = 0.
        float advance  # = 0.
        _RID font_rid
        int font_size  # = 0
        int32_t index  # = 0


cdef extern from "godot_cpp/core/object_id.hpp" namespace "godot" nogil:
    cdef cppclass ObjectID:
        ObjectID()
        ObjectID(const uint64_t)
        ObjectID(const int64_t)

        void operator=(int64_t)
        void operator=(uint64_t)


cdef extern from "godot_cpp/classes/physics_server2d_extension_motion_result.hpp" namespace "godot" nogil:
    cdef struct PhysicsServer2DExtensionMotionResult:
        Vector2 travel
        Vector2 remainder
        Vector2 collision_point
        Vector2 collision_normal
        Vector2 collider_velocity
        real_t collision_depth
        real_t collision_safe_fraction
        real_t collision_unsafe_fraction
        int collision_local_shape
        ObjectID collider_id
        _RID collider
        int collider_shape

ctypedef PhysicsServer2DExtensionMotionResult _PS2DEMotionResult


cdef extern from "godot_cpp/classes/physics_server2d_extension_ray_result.hpp" namespace "godot" nogil:
    cdef struct PhysicsServer2DExtensionRayResult:
        Vector2 position
        Vector2 normal
        _RID rid
        ObjectID collider_id
        GodotCppObject *collider
        int shape

ctypedef PhysicsServer2DExtensionRayResult _PS2DERayResult


cdef extern from "godot_cpp/classes/physics_server2d_extension_shape_rest_info.hpp" namespace "godot" nogil:
    cdef struct PhysicsServer2DExtensionShapeRestInfo:
        Vector2 point
        Vector2 normal
        _RID rid
        ObjectID collider_id
        int shape
        Vector2 linear_velocity

ctypedef PhysicsServer2DExtensionShapeRestInfo _PS2DEShapeRestInfo


cdef extern from "godot_cpp/classes/physics_server2d_extension_shape_result.hpp" namespace "godot" nogil:
    cdef struct PhysicsServer2DExtensionShapeResult:
        _RID rid
        ObjectID collider_id
        GodotCppObject *collider
        int shape

ctypedef PhysicsServer2DExtensionShapeResult _PS2DEShapeResult


cdef extern from "godot_cpp/classes/physics_server3d_extension_motion_collision.hpp" namespace "godot" nogil:
    cdef struct PhysicsServer3DExtensionMotionCollision:
        Vector3 position
        Vector3 normal
        Vector3 collider_velocity
        Vector3 collider_angular_velocity
        real_t depth
        int local_shape
        ObjectID collider_id
        _RID collider
        int collider_shape

ctypedef PhysicsServer3DExtensionMotionCollision _PS3DEMotionCollision


cdef extern from "godot_cpp/classes/physics_server3d_extension_motion_result.hpp" namespace "godot" nogil:
    cdef struct PhysicsServer3DExtensionMotionResult:
        Vector3 travel
        Vector3 remainder
        real_t collision_depth
        real_t collision_safe_fraction
        real_t collision_unsafe_fraction
        _PS3DEMotionCollision collisions[32]
        int collision_count

ctypedef PhysicsServer3DExtensionMotionResult _PS3DEMotionResult


cdef extern from "godot_cpp/classes/physics_server3d_extension_ray_result.hpp" namespace "godot" nogil:
    cdef struct PhysicsServer3DExtensionRayResult:
        Vector3 position
        Vector3 normal
        _RID rid
        ObjectID collider_id
        GodotCppObject *collider
        int shape
        int face_index

ctypedef PhysicsServer3DExtensionRayResult _PS3DERayResult


cdef extern from "godot_cpp/classes/physics_server3d_extension_shape_result.hpp" namespace "godot" nogil:
    cdef struct PhysicsServer3DExtensionShapeRestInfo:
        Vector3 point
        Vector3 normal
        _RID rid
        ObjectID collider_id
        int shape
        Vector3 linear_velocity

ctypedef PhysicsServer3DExtensionShapeRestInfo _PS3DEShapeRestInfo


cdef extern from "godot_cpp/classes/physics_server3d_extension_shape_rest_info.hpp" namespace "godot" nogil:
    cdef struct PhysicsServer3DExtensionShapeResult:
        _RID rid
        ObjectID collider_id
        GodotCppObject *collider
        int shape

ctypedef PhysicsServer3DExtensionShapeResult _PS3DEShapeResult


cdef extern from "godot_cpp/classes/script_language_extension_profiling_info.hpp" namespace "godot" nogil:
    cdef struct ScriptLanguageExtensionProfilingInfo:
        StringName signature
        uint64_t call_count
        uint64_t total_time
        uint64_t self_time

ctypedef ScriptLanguageExtensionProfilingInfo _SLEPInfo
