def as_audio_frame(Pointer pointer):
    return audio_frame_to_pyobject(<const cpp.AudioFrame *>(pointer.ptr))


cdef class AudioFrame:
    def __cinit__(self, float left, float right):
        self.left = left
        self.right = right

    def __repr__(self):
        cls = self.__class__
        return "%s.%s(left=%d, right=%d)" % (cls.__module__, cls.__name__, self.left, self.right)


cdef AudioFrame audio_frame_to_pyobject(const cpp.AudioFrame *af):
    return AudioFrame(af.left, af.right)


cdef int audio_frame_from_pyobject(AudioFrame p_obj, cpp.AudioFrame *r_ret) except -1:
    cdef cpp.AudioFrame ret
    ret.left = p_obj.left
    ret.right = p_obj.right

    r_ret[0] = ret


def as_caret_info(Pointer pointer):
    return caret_info_to_pyobject(<const cpp.CaretInfo *>(pointer.ptr))


cdef class CaretInfo:
    def __cinit__(self, leading_caret, trailing_caret, int leading_direction, int trailing_direction):
        self.data = np.array([leading_caret, trailing_caret], dtype=np.float32).reshape((2, 4))

        self.leading_direction = leading_direction
        self.trailing_direction = trailing_direction

    def __repr__(self):
        cls = self.__class__
        return "%s.%s(leading_caret=%r, trailing_caret=%r, leading_direction=%d, trailing_direction=%d)" % (
            cls.__module__, cls.__name__,
            self.leading_caret, self.trailing_caret,
            self.leading_direction, self.trailing_direction
        )

    @property
    def leading_caret(self):
        return as_rect2(self.data[0])

    @property
    def trailing_caret(self):
        return as_rect2(self.data[1])


cdef CaretInfo caret_info_to_pyobject(const cpp.CaretInfo *ci):
    leading_caret = rect2_to_pyobject(ci.leading_caret)
    trailing_caret = rect2_to_pyobject(ci.trailing_caret)

    return CaretInfo(leading_caret, trailing_caret, ci.leading_direction, ci.trailing_direction)


cdef int caret_info_from_pyobject(CaretInfo p_obj, cpp.CaretInfo *r_ret) except -1:
    cdef cpp.CaretInfo ret
    rect2_from_pyobject(p_obj.data[0], &ret.leading_caret)
    rect2_from_pyobject(p_obj.data[1], &ret.trailing_caret)
    ret.leading_direction = <cpp.TextServer.Direction>p_obj.leading_direction
    ret.trailing_direction = <cpp.TextServer.Direction>p_obj.trailing_direction

    r_ret[0] = ret


def as_glyph(Pointer pointer):
    return glyph_to_pyobject(<const cpp.Glyph *>(pointer.ptr))


cdef class Glyph:
    def __cinit__(self, *, int start=-1, int end=-1, uint8_t count=0, uint8_t repeat=1, uint16_t flags=0,
                  float x_off=0., float y_off=0., float advance=0., RID font_rid=RID(), int font_size=0,
                  int32_t index=0):
        self.start = start
        self.end = end
        self.count = count
        self.repeat = repeat
        self.flags = flags
        self.x_off = x_off
        self.y_off = y_off
        self.font_rid = font_rid
        self.font_size = font_size
        self.index = index

    def __repr__(self):
        cls = self.__class__

        repr_string = "%s.%s(start=%d, end=%d, count=%d, repeat=%d, x_off=%s, y_off=%s, advance=%s, font_rid=%r, " \
                      "font_size=%d, index=%d)"

        return repr_string % (
            cls.__module__,
            cls.__name__,
            self.start,
            self.end,
            self.count,
            self.repeat,
            self.x_off,
            self.y_off,
            self.advance,
            self.font_rid,
            self.font_size,
            self.index
        )


cdef Glyph glyph_to_pyobject(const cpp.Glyph *g):
    return Glyph(start=g.start, end=g.end, count=g.count, repeat=g.repeat, flags=g.flags, x_off=g.x_off, y_off=g.y_off,
                 advance=g.advance, font_rid=rid_to_pyobject(g.font_rid), font_size=g.font_size, index=g.index)


cdef int glyph_from_pyobject(Glyph p_obj, cpp.Glyph *r_ret) except -1:
    cdef cpp.Glyph ret
    ret.start = p_obj.start
    ret.end = p_obj.end
    ret.count = p_obj.count
    ret.repeat = p_obj.repeat
    ret.flags = p_obj.flags
    ret.x_off = p_obj.x_off
    ret.y_off = p_obj.y_off
    ret.advance = p_obj.advance
    rid_from_pyobject(p_obj.font_rid, &ret.font_rid)
    ret.font_size = p_obj.font_size
    ret.index = p_obj.index

    r_ret[0] = ret


def as_object_id(Pointer pointer):
    return object_id_to_pyobject(<const cpp.ObjectID *>(pointer.ptr))


cdef class ObjectID:
    def __cinit__(self, uint64_t id):
        self.id = id

    def __int__(self):
        return self.id

    def __repr__(self):
        cls = self.__class__
        return "%s.%s(id=%s)" % (cls.__module__, cls.__name__, hex(self.id))


cdef object object_id_to_pyobject(const cpp.ObjectID *oid):
    return ObjectID(<uint64_t>oid[0])


cdef int object_id_from_pyobject(object p_obj, cpp.ObjectID *r_ret) except -1:
    cdef uint64_t tmp = int(p_obj)
    cdef cpp.ObjectID oid = cpp.ObjectID(tmp)

    r_ret[0] = oid


def as_physics_server2d_extension_motion_result(Pointer pointer):
    return physics_server2d_extension_motion_result_to_pyobject(<const cpp._PS2DEMotionResult *>(pointer.ptr))


cdef class PhysicsServer2DExtensionMotionResult:
    def __cinit__(self, travel, remainder, collision_point, collision_normal, collider_velocity,
                  float collision_depth, float collision_safe_fraction, float collision_unsafe_fraction,
                  int collision_local_shape, uint64_t collider_id, RID collider, int collider_shape):
        self.data = np.array([
            travel, remainder, collision_point,
            collision_normal, collider_velocity
        ], dtype=np.float32).reshape((5, 2))
        self.collision_depth = collision_depth
        self.collision_safe_fraction = collision_safe_fraction
        self.collision_unsafe_fraction = collision_unsafe_fraction
        self.collision_local_shape = collision_local_shape
        self.collider_id = ObjectID(collider_id)
        self.collider = collider
        self.collider_shape = collider_shape

    def __repr__(self):
        cls = self.__class__

        repr_string = "%s.%s(travel=%r, remainder=%r, collision_point=%r, collistion_normal=%r, collider_velocity=%r" \
                      "collision_depth=%s, collision_save_fraction=%s, collision_unsafe_fraction=%s, " \
                      "collision_local_shape=%d, collider_id=%r, collider=%r, collider_shape=%d"

        return repr_string % (
            cls.__module__,
            cls.__name__,
            self.travel,
            self.remainder,
            self.collision_point,
            self.collision_normal,
            self.collider_velocity,
            self.collision_depth,
            self.collision_safe_fraction,
            self.collision_unsafe_fraction,
            self.collision_local_shape,
            self.collider_id,
            self.collider,
            self.collider_shape
        )

    @property
    def travel(self):
        return as_vector2(self.data[0])

    @property
    def remainder(self):
        return as_vector2(self.data[1])

    @property
    def collision_point(self):
        return as_vector2(self.data[2])

    @property
    def collision_normal(self):
        return as_vector2(self.data[3])

    @property
    def collider_velocity(self):
        return as_vector2(self.data[4])


cdef _PS2DEMotionResult physics_server2d_extension_motion_result_to_pyobject(const cpp._PS2DEMotionResult *mr):
    return PhysicsServer2DExtensionMotionResult(
        vector2_to_pyobject(mr.travel),
        vector2_to_pyobject(mr.remainder),
        vector2_to_pyobject(mr.collision_point),
        vector2_to_pyobject(mr.collision_normal),
        vector2_to_pyobject(mr.collider_velocity),
        mr.collision_depth,
        mr.collision_safe_fraction,
        mr.collision_unsafe_fraction,
        mr.collision_local_shape,
        <uint64_t>mr.collider_id,
        rid_to_pyobject(mr.collider),
        mr.collider_shape
    )


cdef int physics_server2d_extension_motion_result_from_pyobject(_PS2DEMotionResult p_obj,
                                                                cpp._PS2DEMotionResult *r_ret) except -1:
    cdef cpp._PS2DEMotionResult mr
    vector2_from_pyobject(p_obj.data[0], &mr.travel)
    vector2_from_pyobject(p_obj.data[1], &mr.remainder)
    vector2_from_pyobject(p_obj.data[2], &mr.collision_point)
    vector2_from_pyobject(p_obj.data[3], &mr.collision_normal)
    vector2_from_pyobject(p_obj.data[4], &mr.collider_velocity)
    mr.collision_depth = p_obj.collision_depth
    mr.collision_safe_fraction = p_obj.collision_safe_fraction
    mr.collision_unsafe_fraction = p_obj.collision_unsafe_fraction
    mr.collision_local_shape = p_obj.collision_local_shape
    object_id_from_pyobject(p_obj.collider_id, &mr.collider_id)
    rid_from_pyobject(p_obj.collider, &mr.collider)
    mr.collider_shape = p_obj.collider_shape

    r_ret[0] = mr


def as_physics_server2d_extension_ray_result(Pointer pointer):
    return physics_server2d_extension_ray_result_to_pyobject(<const cpp._PS2DERayResult *>(pointer.ptr))


cdef class PhysicsServer2DExtensionRayResult:
    def __cinit__(self, position, normal, RID rid, uint64_t collider_id, Object collider, int shape):
        self.data = np.array([position, normal], dtype=np.float32).reshape((2, 2))
        self.rid = rid
        self.collider_id = ObjectID(collider_id)
        self.collider = collider
        self.shape = shape

    def __repr__(self):
        cls = self.__class__

        return "%s.%s(position=%r, normal=%r, rid=%s, collider_id=%r, collider=%r, shape=%d)" % (
            cls.__module__,
            cls.__name__,
            self.position,
            self.normal,
            self.rid,
            self.collider_id,
            self.collider,
            self.shape
        )

    @property
    def position(self):
        return as_vector2(self.data[0])

    @property
    def normal(self):
        return as_vector2(self.data[1])


cdef _PS2DERayResult physics_server2d_extension_ray_result_to_pyobject(const cpp._PS2DERayResult *rr):
    return PhysicsServer2DExtensionRayResult(
        vector2_to_pyobject(rr.position),
        vector2_to_pyobject(rr.normal),
        rid_to_pyobject(rr.rid),
        <uint64_t>rr.collider_id,
        object_to_pyobject(rr.collider._owner),
        rr.shape
    )

cdef int physics_server2d_extension_ray_result_from_pyobject(_PS2DERayResult p_obj,
                                                             cpp._PS2DERayResult *r_ret) except -1:
    cdef cpp._PS2DERayResult rr
    vector2_from_pyobject(p_obj.data[0], &rr.position)
    vector2_from_pyobject(p_obj.data[1], &rr.normal)
    rid_from_pyobject(p_obj.rid, &rr.rid)
    object_id_from_pyobject(p_obj.collider_id, &rr.collider_id)
    cppobject_from_pyobject(p_obj.collider, &rr.collider)
    rr.shape = p_obj.shape

    r_ret[0] = rr


def as_physics_server2d_extension_shape_rest_info(Pointer pointer):
    return physics_server2d_extension_shape_rest_info_to_pyobject(<const cpp._PS2DEShapeRestInfo *>(pointer.ptr))


cdef class PhysicsServer2DExtensionShapeRestInfo:
    def __cinit__(self, point, normal, RID rid, uint64_t collider_id, int shape, linear_velocity):
        self.data = np.array([point, normal, linear_velocity], dtype=np.float32).reshape((3, 2))
        self.rid = rid
        self.collider_id = ObjectID(collider_id)
        self.shape = shape

    def __repr__(self):
        cls = self.__class__

        return "%s.%s(point=%r, normal=%r, rid=%s, collider_id=%r, shape=%d, linear_velocity=%r)" % (
            cls.__module__,
            cls.__name__,
            self.point,
            self.normal,
            self.rid,
            self.collider_id,
            self.shape,
            self.linear_velocity
        )

    @property
    def point(self):
        return as_vector2(self.data[0])

    @property
    def normal(self):
        return as_vector2(self.data[1])

    @property
    def linear_velocity(self):
        return as_vector2(self.data[2])


cdef _PS2DEShapeRestInfo physics_server2d_extension_shape_rest_info_to_pyobject(const cpp._PS2DEShapeRestInfo *sri):
    return PhysicsServer2DExtensionShapeRestInfo(
        vector2_to_pyobject(sri.point),
        vector2_to_pyobject(sri.normal),
        rid_to_pyobject(sri.rid),
        <uint64_t>sri.collider_id,
        sri.shape,
        vector2_to_pyobject(sri.linear_velocity)
    )

cdef int physics_server2d_extension_shape_rest_info_from_pyobject(_PS2DEShapeRestInfo p_obj,
                                                                  cpp._PS2DEShapeRestInfo *r_ret) except -1:
    cdef cpp._PS2DEShapeRestInfo sri
    vector2_from_pyobject(p_obj.data[0], &sri.point)
    vector2_from_pyobject(p_obj.data[1], &sri.normal)
    rid_from_pyobject(p_obj.rid, &sri.rid)
    object_id_from_pyobject(p_obj.collider_id, &sri.collider_id)
    sri.shape = p_obj.shape
    vector2_from_pyobject(p_obj.data[2], &sri.linear_velocity)

    r_ret[0] = sri


def as_physics_server2d_extension_shape_result(Pointer pointer):
    return physics_server2d_extension_shape_result_to_pyobject(<const cpp._PS2DEShapeResult *>(pointer.ptr))


cdef class PhysicsServer2DExtensionShapeResult:
    def __cinit__(self, RID rid, uint64_t collider_id, Object collider, int shape):
        self.rid = rid
        self.collider_id = ObjectID(collider_id)
        self.collider = collider
        self.shape = shape

    def __repr__(self):
        cls = self.__class__

        return "%s.%s(rid=%s, collider_id=%r, collider=%r, shape=%d)" % (
            cls.__module__,
            cls.__name__,
            self.rid,
            self.collider_id,
            self.collider,
            self.shape
        )

cdef _PS2DEShapeResult physics_server2d_extension_shape_result_to_pyobject(const cpp._PS2DEShapeResult *sr):
    return PhysicsServer2DExtensionShapeResult(rid_to_pyobject(sr.rid), <uint64_t>sr.collider_id,
                                               object_to_pyobject(sr.collider._owner), sr.shape)

cdef int physics_server2d_extension_shape_result_from_pyobject(_PS2DEShapeResult p_obj,
                                                               cpp._PS2DEShapeResult *r_ret) except -1:
    cdef cpp._PS2DEShapeResult sr
    rid_from_pyobject(p_obj.rid, &sr.rid)
    object_id_from_pyobject(p_obj.collider_id, &sr.collider_id)
    cppobject_from_pyobject(p_obj.collider, &sr.collider)
    sr.shape = p_obj.shape

    r_ret[0] = sr


def as_physics_server3d_extension_motion_collision(Pointer pointer):
    return physics_server3d_extension_motion_collision_to_pyobject(<const cpp._PS3DEMotionCollision *>(pointer.ptr))


cdef class PhysicsServer3DExtensionMotionCollision:
    def __cinit__(self, position, normal, collider_velocity, collider_angular_velocity, float depth,
                  int local_shape, uint64_t collider_id, RID collider, int collider_shape):
        self.data = np.array([
            position, normal, collider_velocity, collider_angular_velocity
        ], dtype=np.float32).reshape((4, 3))
        self.depth = depth
        self.local_shape = local_shape
        self.collider_id = ObjectID(collider_id)
        self.collider = collider
        self.collider_shape = collider_shape

    def __repr__(self):
        cls = self.__class__

        repr_string = "%s.%s(position=%r, normal=%r, collider_velocity=%r, collider_angular_velocity-%r, depth=%s, " \
                      "local_shape=%d, collider_id=%r, collider=%r, collider_shape=%d"

        return repr_string % (
            cls.__module__,
            cls.__name__,
            self.position,
            self.normal,
            self.collider_velocity,
            self.collider_angular_velocity,
            self.depth,
            self.local_shape,
            self.collider_id,
            self.collider,
            self.collider_shape
        )


    @property
    def position(self):
        return as_vector3(self.data[0])

    @property
    def normal(self):
        return as_vector3(self.data[1])

    @property
    def collider_velocity(self):
        return as_vector3(self.data[2])

    @property
    def collider_angular_velocity(self):
        return as_vector3(self.data[3])


cdef _PS3DEMotionCollision physics_server3d_extension_motion_collision_to_pyobject(const cpp._PS3DEMotionCollision *mc):
    return PhysicsServer3DExtensionMotionCollision(
        vector3_to_pyobject(mc.position),
        vector3_to_pyobject(mc.normal),
        vector3_to_pyobject(mc.collider_velocity),
        vector3_to_pyobject(mc.collider_angular_velocity),
        mc.depth, mc.local_shape, <uint64_t>mc.collider_id,
        rid_to_pyobject(mc.collider),
        mc.collider_shape
    )

cdef int physics_server3d_extension_motion_collision_from_pyobject(_PS3DEMotionCollision p_obj,
                                                                   cpp._PS3DEMotionCollision *r_ret) except -1:
    cdef cpp._PS3DEMotionCollision mc
    vector3_from_pyobject(p_obj.data[0], &mc.position)
    vector3_from_pyobject(p_obj.data[1], &mc.normal)
    vector3_from_pyobject(p_obj.data[2], &mc.collider_velocity)
    vector3_from_pyobject(p_obj.data[3], &mc.collider_angular_velocity)
    mc.depth = p_obj.depth
    mc.local_shape = p_obj.local_shape
    object_id_from_pyobject(p_obj.collider_id, &mc.collider_id)
    rid_from_pyobject(p_obj.collider, &mc.collider)
    mc.collider_shape = p_obj.collider_shape

    r_ret[0] = mc


def as_physics_server3d_extension_motion_result(Pointer pointer):
    return physics_server3d_extension_motion_result_to_pyobject(<const cpp._PS3DEMotionResult *>(pointer.ptr))


cdef class PhysicsServer3DExtensionMotionResult:
    def __cinit__(self, travel, remainder, float collision_depth, float collision_safe_fraction,
                  float collision_unsafe_fraction, collisions: List[PhysicsServer3DExtensionMotionCollision]):
        self.data = np.array([travel, remainder], dtype=np.float32).reshape((2, 3))
        self.collision_depth = collision_depth
        self.collision_safe_fraction = collision_safe_fraction
        self.collision_unsafe_fraction = collision_unsafe_fraction
        self.collisions = collisions

    def __repr__(self):
        cls = self.__class__

        repr_string = "%s.%s(travel=%r, remainder=%r, collision_depth=%s, collision_save_fraction=%s, " \
                      "collision_unsafe_fraction=%s, collisions=%r)"

        return repr_string % (
            cls.__module__,
            cls.__name__,
            self.travel,
            self.remainder,
            self.collision_depth,
            self.collision_safe_fraction,
            self.collision_unsafe_fraction,
            self.collisions
        )

    @property
    def travel(self):
        return as_vector3(self.data[0])

    @property
    def remainder(self):
        return as_vector3(self.data[1])


cdef _PS3DEMotionResult physics_server3d_extension_motion_result_to_pyobject(const cpp._PS3DEMotionResult *mr):
    cdef list collisions = PyList_New(mr.collision_count)
    cdef size_t i
    cdef object motion_collision
    for i in range(mr.collision_count):
        motion_collision = physics_server3d_extension_motion_collision_to_pyobject(&mr.collisions[i])
        ref.Py_INCREF(motion_collision)
        PyList_SET_ITEM(collisions, i, motion_collision)

    return PhysicsServer3DExtensionMotionResult(
        vector3_to_pyobject(mr.travel),
        vector3_to_pyobject(mr.remainder),
        mr.collision_depth,
        mr.collision_safe_fraction,
        mr.collision_unsafe_fraction,
        collisions
    )

cdef int physics_server3d_extension_motion_result_from_pyobject(_PS3DEMotionResult p_obj,
                                                                cpp._PS3DEMotionResult *r_ret) except -1:
    cdef size_t i
    cdef cpp._PS3DEMotionResult mr
    vector3_from_pyobject(p_obj.data[0], &mr.travel)
    vector3_from_pyobject(p_obj.data[0], &mr.remainder)
    mr.collision_depth = p_obj.collision_depth
    mr.collision_safe_fraction = p_obj.collision_safe_fraction
    mr.collision_unsafe_fraction = p_obj.collision_unsafe_fraction
    mr.collision_count = len(p_obj.collisions)

    for i in range(mr.collision_count):
        physics_server3d_extension_motion_collision_from_pyobject(p_obj.collisions[i], &mr.collisions[i])

    r_ret[0] = mr


def as_physics_server3d_extension_ray_result(Pointer pointer):
    return physics_server3d_extension_ray_result_to_pyobject(<const cpp._PS3DERayResult *>(pointer.ptr))


cdef class PhysicsServer3DExtensionRayResult:
    def __cinit__(self, position, normal, RID rid, uint64_t collider_id, Object collider, int shape, int face_index):
        self.data = np.array([position, normal], dtype=np.float32).reshape((2, 3))
        self.rid = rid
        self.collider_id = ObjectID(collider_id)
        self.collider = collider
        self.shape = shape
        self.face_index = face_index

    def __repr__(self):
        cls = self.__class__

        return "%s.%s(position=%r, normal=%r, rid=%s, collider_id=%r, collider=%r, shape=%d, face_index=%d)" % (
            cls.__module__,
            cls.__name__,
            self.position,
            self.normal,
            self.rid,
            self.collider_id,
            self.collider,
            self.shape,
            self.face_index
        )

    @property
    def position(self):
        return as_vector3(self.data[0])

    @property
    def normal(self):
        return as_vector3(self.data[1])


cdef _PS3DERayResult physics_server3d_extension_ray_result_to_pyobject(const cpp._PS3DERayResult *rr):
    return PhysicsServer3DExtensionRayResult(
        vector3_to_pyobject(rr.position),
        vector3_to_pyobject(rr.normal),
        rid_to_pyobject(rr.rid),
        <uint64_t>rr.collider_id,
        object_to_pyobject(rr.collider._owner),
        rr.shape,
        rr.face_index
    )

cdef int physics_server3d_extension_ray_result_from_pyobject(_PS3DERayResult p_obj,
                                                             cpp._PS3DERayResult *r_ret) except -1:
    cdef cpp._PS3DERayResult rr
    vector3_from_pyobject(p_obj.data[0], &rr.position)
    vector3_from_pyobject(p_obj.data[1], &rr.normal)
    rid_from_pyobject(p_obj.rid, &rr.rid)
    object_id_from_pyobject(p_obj.collider_id, &rr.collider_id)
    cppobject_from_pyobject(p_obj.collider, &rr.collider)
    rr.shape = p_obj.shape
    rr.face_index = p_obj.face_index

    r_ret[0] = rr


def as_physics_server3d_extension_shape_rest_info(Pointer pointer):
    return physics_server3d_extension_shape_rest_info_to_pyobject(<const cpp._PS3DEShapeRestInfo *>(pointer.ptr))


cdef class PhysicsServer3DExtensionShapeRestInfo:
    def __cinit__(self, point, normal, RID rid, uint64_t collider_id, int shape, linear_velocity):
        self.data = np.array([point, normal, linear_velocity], dtype=np.float32).reshape((3, 3))
        self.rid = rid
        self.collider_id = ObjectID(collider_id)
        self.shape = shape

    def __repr__(self):
        cls = self.__class__

        return "%s.%s(point=%r, normal=%r, rid=%s, collider_id=%r, shape=%d, linear_velocity=%r)" % (
            cls.__module__,
            cls.__name__,
            self.point,
            self.normal,
            self.rid,
            self.collider_id,
            self.shape,
            self.linear_velocity
        )

    @property
    def point(self):
        return as_vector3(self.data[0])

    @property
    def normal(self):
        return as_vector3(self.data[1])

    @property
    def linear_velocity(self):
        return as_vector3(self.data[2])


cdef _PS3DEShapeRestInfo physics_server3d_extension_shape_rest_info_to_pyobject(const cpp._PS3DEShapeRestInfo *sri):
    return PhysicsServer3DExtensionShapeRestInfo(
        vector3_to_pyobject(sri.point),
        vector3_to_pyobject(sri.normal),
        rid_to_pyobject(sri.rid),
        <uint64_t>sri.collider_id,
        sri.shape,
        vector3_to_pyobject(sri.linear_velocity)
    )

cdef int physics_server3d_extension_shape_rest_info_from_pyobject(_PS3DEShapeRestInfo p_obj,
                                                                  cpp._PS3DEShapeRestInfo *r_ret) except -1:
    cdef cpp._PS3DEShapeRestInfo sri
    vector3_from_pyobject(p_obj.data[0], &sri.point)
    vector3_from_pyobject(p_obj.data[1], &sri.normal)
    rid_from_pyobject(p_obj.rid, &sri.rid)
    object_id_from_pyobject(p_obj.collider_id, &sri.collider_id)
    sri.shape = p_obj.shape
    vector3_from_pyobject(p_obj.data[2], &sri.linear_velocity)

    r_ret[0] = sri


def as_physics_server3d_extension_shape_result(Pointer pointer):
    return physics_server3d_extension_shape_result_to_pyobject(<const cpp._PS3DEShapeResult *>(pointer.ptr))


cdef class PhysicsServer3DExtensionShapeResult:
    def __cinit__(self, RID rid, uint64_t collider_id, Object collider, int shape):
        self.rid = rid
        self.collider_id = ObjectID(collider_id)
        self.collider = collider
        self.shape = shape

    def __repr__(self):
        cls = self.__class__

        return "%s.%s(rid=%s, collider_id=%r, collider=%r, shape=%d)" % (
            cls.__module__,
            cls.__name__,
            self.rid,
            self.collider_id,
            self.collider,
            self.shape
        )

cdef _PS3DEShapeResult physics_server3d_extension_shape_result_to_pyobject(const cpp._PS3DEShapeResult *sr):
    return PhysicsServer3DExtensionShapeResult(
        rid_to_pyobject(sr.rid),
        <uint64_t>sr.collider_id,
        object_to_pyobject(sr.collider._owner),
        sr.shape
    )

cdef int physics_server3d_extension_shape_result_from_pyobject(_PS3DEShapeResult p_obj,
                                                             cpp._PS3DEShapeResult *r_ret) except -1:
    cdef cpp._PS3DEShapeResult sr
    rid_from_pyobject(p_obj.rid, &sr.rid)
    object_id_from_pyobject(p_obj.collider_id, &sr.collider_id)
    cppobject_from_pyobject(p_obj.collider, &sr.collider)
    sr.shape = p_obj.shape

    r_ret[0] = sr


def as_script_language_extension_profiling_info(Pointer pointer):
    return script_language_extension_profiling_info_to_pyobject(<const cpp._SLEPInfo *>(pointer.ptr))


cdef class ScriptLanguageExtensionProfilingInfo:
    def __cinit__(self, signature, call_count, total_time, self_time):
        self.signature = str(signature)
        self.call_count = call_count
        self.total_time = total_time
        self.self_time = self_time

    def __repr__(self):
        cls = self.__class__

        return "%s.%s(signature=%s, call_count=%d, total_time=%d, self_time=%d)" % (
            cls.__module__,
            cls.__name__,
            self.signature,
            self.call_count,
            self.total_time,
            self.self_time
        )


cdef _SLEPInfo script_language_extension_profiling_info_to_pyobject(const cpp._SLEPInfo *p_info):
    return ScriptLanguageExtensionProfilingInfo(
        string_name_to_pyobject(p_info.signature),
        p_info.call_count,
        p_info.total_time,
        p_info.self_time
    )

cdef int script_language_extension_profiling_info_from_pyobject(_SLEPInfo p_obj, cpp._SLEPInfo *r_ret):
    cdef cpp.ScriptLanguageExtensionProfilingInfo info
    string_name_from_pyobject(p_obj.signature, &info.signature)
    info.call_count = p_obj.call_count
    info.total_time = p_obj.total_time
    info.self_time = p_obj.self_time

    r_ret[0] = info
