from libc.stdint cimport uint64_t, int64_t
from libc.stddef cimport wchar_t

from godot_headers.gdnative_api cimport *

cdef extern from "Defs.hpp":
    ctypedef float real_t


cdef extern from "Defs.hpp" namespace "godot":
    cdef enum Error:
        OK
        FAILED  # Generic fail error
        ERR_UNAVAILABLE  # What is requested is unsupported/unavailable
        ERR_UNCONFIGURED  # The object being used hasnt been properly set up yet
        ERR_UNAUTHORIZED  # Missing credentials for requested resource
        ERR_PARAMETER_RANGE_ERROR  # Parameter given out of range (5)
        ERR_OUT_OF_MEMORY  # Out of memory
        ERR_FILE_NOT_FOUND
        ERR_FILE_BAD_DRIVE
        ERR_FILE_BAD_PATH
        ERR_FILE_NO_PERMISSION  # 10
        ERR_FILE_ALREADY_IN_USE
        ERR_FILE_CANT_OPEN
        ERR_FILE_CANT_WRITE
        ERR_FILE_CANT_READ
        ERR_FILE_UNRECOGNIZED  # 15
        ERR_FILE_CORRUPT
        ERR_FILE_MISSING_DEPENDENCIES
        ERR_FILE_EOF
        ERR_CANT_OPEN  # Can't open a resource/socket/file
        ERR_CANT_CREATE  # 20
        ERR_QUERY_FAILED
        ERR_ALREADY_IN_USE
        ERR_LOCKED  # Resource is locked
        ERR_TIMEOUT,
        ERR_CANT_CONNECT  # 25
        ERR_CANT_RESOLVE
        ERR_CONNECTION_ERROR
        ERR_CANT_AQUIRE_RESOURCE
        ERR_CANT_FORK
        ERR_INVALID_DATA  # Data passed is invalid   (30)
        ERR_INVALID_PARAMETER  # Parameter passed is invalid
        ERR_ALREADY_EXISTS  # When adding, item already exists
        ERR_DOES_NOT_EXIST  # When retrieving/erasing, it item does not exist
        ERR_DATABASE_CANT_READ  # Database is full
        ERR_DATABASE_CANT_WRITE  # Database is full  (35)
        ERR_COMPILATION_FAILED
        ERR_METHOD_NOT_FOUND
        ERR_LINK_FAILED
        ERR_SCRIPT_FAILED
        ERR_CYCLIC_LINK  # 40
        ERR_INVALID_DECLARATION
        ERR_DUPLICATE_SYMBOL
        ERR_PARSE_ERROR
        ERR_BUSY
        ERR_SKIP  # 45
        ERR_HELP  # User requested help!!
        ERR_BUG  # A bug in the software certainly happened, due to a double check failing or unexpected behavior.
        ERR_PRINTER_ON_FIRE  # The parallel port printer is engulfed in flames
        ERR_OMFG_THIS_IS_VERY_VERY_BAD  # Shit happens, has never been used, though
        ERR_WTF = ERR_OMFG_THIS_IS_VERY_VERY_BAD  # Short version of the above


cdef extern from "AABB.hpp" namespace "godot" nogil:
    cdef cppclass Vector3
    cdef cppclass String

    cdef cppclass AABB:
        Vector3 position
        Vector3 size

        real_t get_area()
        bint has_no_surface()
        const Vector3 &get_position
        void set_position(const Vector3 &position)
        const Vector3 &get_size()
        void set_size(const Vector3 &size)

        bint operator==(const AABB&)
        bint operator!=(const AABB&)

        bint intersects(const AABB&)  # Both AABBs overlap
        bint intersects_inclusive(const AABB&)  #  Both AABBs (or their faces) overlap
        bint encloses(const AABB&)  # Other AABB is completely inside this

        AABB merge(const AABB&)
        void merge_with(const AABB&)  # Merge with another AABB
        AABB intersection(const AABB&)  # Get box where two intersect, empty if no intersection occurs
        bint intersects_segment(const Vector3 &from_, const Vector3 &to, Vector3 *clip=nullptr, Vector3 *normal=nullptr)
        bint intersects_ray(const Vector3 &from_, const Vector3 &dir, Vector3 *clip=nullptr, Vector3 *normal=nullptr)
        bint smits_inersects_ray(const Vector3 &from_, const Vector3 &dir, real_t t0, real_t t1)

        bint intersects_convex_shape(const Plane *plane, int plane_count)
        bint intersects_plane(const Plane&)
        bint has_point(const Vector3&)
        Vector3 get_support(const Vector3&)

        Vector3 get_longest_axis()
        int get_longest_axis_index()
        real_t get_longest_axis_size()

        Vector3 get_shortest_axis()
        int get_shortest_axis_index()
        real_t get_shortest_axis_size()

        AABB grow(real_t by)
        void grow_by(real_t amount)

        void get_edge(int edge, Vector3 &from_, Vector3 &to)
        Vector3 get_endpoint(int point)

        AABB expand(const Vector3 &vector)
        void project_range_in_plane(const Plane &plane, real_t &min, real_t &max)
        void expand_to(const Vector3 &vector)

        # String operator String()

        AABB() except +
        AABB(const Vector3 &pos, const Vector3 &size) except+

        object wrap "pythonize" ()


cdef extern from "Object.hpp" namespace "godot" nogil:
    cdef cppclass __Object "godot::Object"


cdef extern from "Array.hpp" namespace "godot" nogil:
    cdef cppclass Variant
    cdef cppclass PoolByteArray
    cdef cppclass PoolIntArray
    cdef cppclass PoolRealArray
    cdef cppclass PoolStringArray
    cdef cppclass PoolVector2Array
    cdef cppclass PoolVector3Array
    cdef cppclass PoolColorArray

    cdef cppclass Array:
        Array() except+
        Array(const Array &other) except+

        Array(const PoolByteArray &a) except+
        Array(const PoolIntArray &a) except+
        Array(const PoolRealArray &a) except+
        Array(const PoolStringArray &a) except+
        Array(const PoolVector2Array &a) except+
        Array(const PoolVector3Array &a) except+
        Array(const PoolColorArray &a) except+

        Array(object) except+

        @staticmethod
        Array make(...) except +

        Variant& operator[](const int idx)
        Variant operator[](const int idx)

        void append(const Variant &v)
        void append(object)
        void clear()
        int count(const Variant &v)
        int count(object)
        bint empty()
        void erase(const Variant &v)
        void erase(object)
        Variant front()
        object front()
        Variant back()
        object back()
        int find(const Variant &what, const int from_=0)
        int find(object what, const int from_=0)
        int find_last(const Variant &what)
        int find_last(object what)
        bint has(const Variant &what)
        bint has(object what)
        uint32_t hash()
        void insert(const int pos, const Variant &value)
        void insert(const int pos, object value)
        void invert()
        bint is_shared()
        Variant pop_back()
        Variant pop_front()
        void push_back(const Variant&)
        void push_back(object)
        void push_front(const Variant&)
        void push_front(object)
        void remove(const int idx)
        int size()
        void resize(const int size)
        int rfind(const Variant &what, const int from_=0)
        int rfind(object what, const int from_=0)
        void sort()
        void sort_custom(__Object *obj, const String &func)
        int bsearch(const Variant &value, const bint before=True)
        int bsearch(object value, const bint before=True)
        Array duplicate(const bint deep=False)
        Variant max()
        object max()
        Variant min()
        object min()
        void shuffle()

        object wrap "pythonize" ()


cdef extern from "Basis.hpp" namespace "godot" nogil:
    cdef cppclass Basis
    cdef cppclass Quat

    cdef cppclass ColumnVector3[column]:
        enum Axis:
            AXIS_X
            AXIS_Y
            AXIS_Z
        ColumnVector3(const ColumnVector3[column] *value) except+
        # Vector3 operator Vector3()
        const real_t& operator[](int axis)
        real_t& operator[](int axis)

        # ColumnVector3[column]& operator+=(const Vector3&)
        Vector3 operator+(const Vector3&)
        # ColumnVector3[column]& operator-=(const Vector3&)
        Vector3 operator-(const Vector3&)
        # *=
        Vector3 operator*(const Vector3&)
        # /=
        Vector3 operator/(const Vector3&)
        # *= scalar
        Vector3 operator*(real_t)
        # /= scalar
        Vector3 operator/(real_t)
        Vector3 operator-()

        bint operator==(const Vector3&)
        bint operator!=(const Vector3&)
        bint operator<(const Vector3&)
        bint operator<=(const Vector3&)

        Vector3 abs()
        Vector3 ceil()
        Vector3 cross(const Vector3 &b)
        Vector3 linear_interpolate(const Vector3 &b, real_t t)
        Vector3 cubic_interpolate(const Vector3 &b, const Vector3 &pre_a, const Vector3 &post_b, const real_t t)
        Vector3 bounce(const Vector3 &normal)
        real_t length()
        real_t length_squared()
        real_t distance_squared_to(const Vector3 &b)
        real_t distance_to(const Vector3 &b)
        real_t dot(const Vector3 &b)
        real_t angle_to(const Vector3 &b)
        Vector3 floor()
        Vector3 inverse()
        bint is_normalized()
        Basis outer(const Vector3 &b)
        int max_axis()
        int min_axis()
        void normalize()
        Vector3 normalized()
        Vector3 reflect(Vector3 &by)
        Vector3 rotated(const Vector3 &axis, real_t phi)
        void rotate(const Vector3 &axis, real_t phi)
        Vector3 slide(const Vector3 &by)
        void snap(real_t val)
        Vector3 snapped(const float by)

        # String operator String()

        object wrap "pythonize" ()

    cdef cppclass Basis:
        # ColumnVector3[0] x
        # ColumnVector3[1] y
        # ColumnVector3[2] z

        ColumnVector3 x
        ColumnVector3 y
        ColumnVector3 z

        Basis(const Basis &basis) except +
        Basis(const Quat &quat) except +
        Basis(const Vector3 &euler) except +
        Basis(const Vector3 &axis, real_t phi) except +
        Basis(const Vector3 &row0, const Vector3 &row1, const Vector3 &row2) except +
        Basis(real_t xx, real_t xy, real_t xz,
              real_t yx, real_t yy, real_t yz,
              real_t zx, real_t zy, real_t zz) except +
        Basis() except+

        const Vector3 operator[](int axis)
        ColumnVector3& operator[](int axis)

        void invert()
        bint isequal_approx(const Basis &a, const Basis &b)
        bint is_orthogonal()
        bint is_rotation()
        void transpose()
        Basis inverse()
        Basis transposed()
        real_t determinant()
        Vector3 get_axis(int axis)
        void set_axis(int axis, const Vector3 &value)
        void rotate(const Vector3 &axis, real_t phi)
        Basis rotated(const Vector3 &axis, real_t phi)
        void scale(const Vector3 &scale)
        Basis scaled(const Vector3 &scale)
        Vector3 get_scale()

        Basis slerp(Basis b, real_t t)

        Vector3 get_euler_xyz()
        void set_euler_xyz(const Vector3 &euler)
        void set_euler_yxz(const Vector3 &euler)
        Vector3 get_euler_yxz()
        Vector3 get_euler()
        void set_euler(const Vector3 &euler)

        real_t tdotx(const Vector3&)
        real_t tdoty(const Vector3&)
        real_t tdotz(const Vector3&)

        bint operator==(const Basis&)
        bint operator!=(const Basis&)

        Vector3 xform(const Vector3 &vector)
        Vector3 xform_inv(const Vector3 &vector)

        # *=
        Basis operator*(const Basis&)
        # +=
        Basis operator+(const Basis&)
        # -=
        Basis operator-(const Basis&)
        # *= scalar
        Basis operator*(real_t val)

        int get_orthogonal_index()
        void set_orthogonal_index(int index)

        # String operator String()

        void get_axis_and_angle(Vector3 &axis, real_t &angle)
        void set(real_t xx, real_t xy, real_t xz,
                 real_t yx, real_t yy, real_t yz,
                 real_t zx, real_t zy, real_t zz)
        Vector3 get_column(int i)
        Vector3 get_row(int i)
        Vector3 get_main_diagonal()
        void set_row(int i, const Vector3 &p_row)
        Basis transpose_xform(const Basis &m)
        bint is_symmetric()
        Basis diagonalize()

        # Quat operator Quat()
        object wrap "pythonize" ()


cdef extern from "Color.hpp" namespace "godot" nogil:
    cdef cppclass Color:  # C++ struct
        float r, g, b, a

        bint operator==(const Color&)
        bint operator!=(const Color&)

        uint32_t to_32()
        uint32_t to_ARGB32()
        uint32_t to_ABGR32()
        uint64_t to_ABGR64()
        uint64_t to_ARGB64()
        uint32_t to_RGBA32()
        uint64_t to_RGBA64()
        float gray()
        uint8_t get_r8()
        uint8_t get_g8()
        uint8_t get_b8()
        uint8_t get_a8()
        float get_h()
        float get_s()
        float get_v()
        void set_hsv(float, float, float, float alpha=1.0)

        Color darkened(const float amount)
        Color lightened(const float amount)
        Color from_hsv(float h, float, s, float v, float a=1.0)

        float& operator[](int)
        const float& operator[](int)

        void invert()
        void contrast()

        Color inverted()
        Color contrasted()
        Color linear_interpolate(const Color&, float)
        Color blend(const Color&)
        Color to_linear()

        @staticmethod
        Color hex(uint32_t) except +

        @staticmethod
        Color html(const String&) except +

        String to_html(bint alpha=1.0)

        @staticmethod
        bint html_is_valid(const String&)

        bint operator<(const Color&)

        Color()
        Color(float, float, float)
        Color(float, float, float, float)

        object wrap "pythonize" ()


cdef extern from "Dictionary.hpp" namespace "godot" nogil:
    cdef cppclass Dictionary:
        Dictionary() except +
        Dictionary(const Dictionary &other) except +

        @staticmethod
        Dictionary make(...) except +

        void clear()
        bint empty()
        void erase(const Variant &key)
        bool has(const Variant &key)
        bool has_all(const Array &keys)
        uint32_t hash()
        Array keys()
        Variant& operator[](const Variant &key)
        const Variant& operator[](const Variant &key)
        int size()
        String to_json()
        Array values()

        object wrap "pythonize" ()


cdef extern from "NodePath.hpp" namespace "godot" nogil:
    cdef cppclass NodePath:
        NodePath() except +
        NodePath(const NodePath&) except +
        NodePath(const String&) except +
        NodePath(const char *) except +

        String get_name(const int idx)
        int get_name_count()
        String get_subname(const int idx)
        int get_subname_count()
        bint is_absolute()
        bint is_empty()
        NodePath get_as_property_path()
        String get_concatenated_subnames()

        # String operator String()

        bint operator==(const NodePath&)

        object wrap "pythonize" ()


cdef extern from "Plane.hpp" namespace "godot" nogil:
    cdef enum ClockDirection:
        CLOCKWISE
        COUNTERCLOCKWISE

    cdef cppclass Plane:
        Vector3 normal
        real_t d

        set_normal(const Vector3&)
        Vector3 get_normal()
        void normalize()
        Plane normalized()

        # Plane-Point operations
        Vector3 center()
        Vector3 get_any_point()
        Vector3 get_any_perpendicular_normal()
        bint is_point_over(const Vector3&)
        real_t distance_to(const Vector3&)
        bint has_point(const Vector3&, real_t _epsilon=CMP_EPSILON)

        # Intersections
        bint intersect_3(const Plane&, const Plane&, Vector3 *result=0)
        bint intersects_ray(Vector3 from_, Vector3 dir, Vector3 *intersection)
        bint intersects_segment(Vector3 begin, Vector3 end, Vector3 *intersection)

        Vector3 project(const Vector3 &point)

        # Misc
        Plane operator-()
        bint is_almost_like(const Plane&)
        bint operator==(const Plane&)
        bint operator!=(const Plane&)

        # String operator String()

        Plane() except +
        Plane(real_t a, real_t b, real_t c, real_t d) except +
        Plane(const Vector3 &normal, real_t d) except +
        Plane(const Vector3 &point, const Vector3 &normal) except +
        Plane(const Vector3 &point1, const Vector3 &point2,
              const Vector3 &point3, ClockDirection dir=CLOCKWISE) except +

        object wrap "pythonize" ()


cdef extern from * namespace "godot":
    cdef cppclass Vector2

cdef extern from "PoolArrays.hpp" namespace "godot" nogil:
    cdef cppclass PoolByteArray:
        cppclass Read:
            const uint8_t *ptr()
            const uint8_t& operator[](int)

        cppclass Write:
            uint8_t *ptr()
            uint8_t& operator[](int)

        PoolByteArray() except +
        PoolByteArray(const PoolByteArray&) except +
        PoolByteArray(const Array&) except +
        Read read() except +
        Write write() except +
        void append(const uint8_t data)
        void append_array(const PoolByteArray &array)
        int insert(const int idx, const uint8_t data)
        void invert()
        void push_back(const uint8_t data)
        void remove(const int idx)
        void resize(const int size)
        void set(const int idx, const uint8_t data)
        uint8_t operator[](const int idx)
        int size()

        object wrap "pythonize" ()


    cdef cppclass PoolIntArray:
        cppclass Read:
            const int *ptr()
            const int& operator[](int)

        cppclass Write:
            int *ptr()
            int& operator[](int)

        PoolIntArray() except +
        PoolIntArray(const PoolIntArray&) except +
        PoolIntArray(const Array&) except +
        Read read() except +
        Write write() except +
        void append(const int data)
        void append_array(const PoolIntArray &array)
        int insert(const int idx, const int data)
        void invert()
        void push_back(const int data)
        void remove(const int idx)
        void resize(const int size)
        void set(const int idx, const int data)
        int operator[](const int idx)
        int size()

        object wrap "pythonize" ()


    cdef cppclass PoolRealArray:
        cppclass Read:
            const real_t *ptr()
            const real_t& operator[](int)

        cppclass Write:
            real_t *ptr()
            real_t& operator[](int)

        PoolRealArray() except +
        PoolRealArray(const PoolRealArray&) except +
        PoolRealArray(const Array&) except +
        Read read() except +
        Write write() except +
        void append(const real_t data)
        void append_array(const PoolRealArray &array)
        int insert(const int idx, const real_t data)
        void invert()
        void push_back(const real_t data)
        void remove(const int idx)
        void resize(const int size)
        void set(const int idx, const real_t data)
        real_t operator[](const int idx)
        int size()

        object wrap "pythonize" ()


    cdef cppclass PoolStringArray:
        cppclass Read:
            const String *ptr()
            const String& operator[](int)

        cppclass Write:
            String *ptr()
            String& operator[](int)

        PoolStringArray() except +
        PoolStringArray(const PoolStringArray&) except +
        PoolStringArray(const Array&) except +
        Read read() except +
        Write write() except +
        void append(const String &data)
        void append_array(const PoolStringArray &array)
        int insert(const int idx, const String &data)
        void invert()
        void push_back(const String &data)
        void remove(const int idx)
        void resize(const int size)
        void set(const int idx, const String &data)
        const String operator[](const int idx)
        int size()

        object wrap "pythonize" ()


    cdef cppclass PoolVector2Array:
        cppclass Read:
            const Vector2 *ptr()
            const Vector2& operator[](int)

        cppclass Write:
            Vector2 *ptr()
            Vector2& operator[](int)

        PoolVector2Array() except +
        PoolVector2Array(const PoolVector2Array&) except +
        PoolVector2Array(const Array&) except +
        Read read() except +
        Write write() except +
        void append(const Vector2 &data)
        void append_array(const PoolVector2Array &array)
        int insert(const int idx, const Vector2 &data)
        void invert()
        void push_back(const Vector2 &data)
        void remove(const int idx)
        void resize(const int size)
        void set(const int idx, const Vector2 &data)
        const Vector2 operator[](const int idx)
        int size()

        object wrap "pythonize" ()


    cdef cppclass PoolVector3Array:
        cppclass Read:
            const Vector3 *ptr()
            const Vector3& operator[](int)

        cppclass Write:
            Vector3 *ptr()
            Vector3& operator[](int)

        PoolVector3Array() except +
        PoolVector3Array(const PoolVector3Array&) except +
        PoolVector3Array(const Array&) except +
        Read read() except +
        Write write() except +
        void append(const Vector3 &data)
        void append_array(const PoolVector3Array &array)
        int insert(const int idx, const Vector3 &data)
        void invert()
        void push_back(const Vector3 &data)
        void remove(const int idx)
        void resize(const int size)
        void set(const int idx, const Vector3 &data)
        const Vector3 operator[](const int idx)
        int size()

        object wrap "pythonize" ()


    cdef cppclass PoolColorArray:
        cppclass Read:
            const Color *ptr()
            const Color& operator[](int)

        cppclass Write:
            Color *ptr()
            Color& operator[](int)

        PoolColorArray() except +
        PoolColorArray(const PoolColorArray&) except +
        PoolColorArray(const Array&) except +
        Read read() except +
        Write write() except +
        void append(const Color &data)
        void append_array(const PoolColorArray &array)
        int insert(const int idx, const Color &data)
        void invert()
        void push_back(const Color &data)
        void remove(const int idx)
        void resize(const int size)
        void set(const int idx, const Color &data)
        const Color operator[](const int idx)
        int size()

        object wrap "pythonize" ()


cdef extern from "Quat.hpp" namespace "godot" nogil:
    cdef cppclass Quat:
        real_t x, y, z, w

        real_t length_squared()
        real_t length()
        void normalize()
        Quat normalized()
        bint is_normalized()
        Quat inverse()
        void set_euler_xyz(const Vector3 &euler)
        Vector3 get_euler_xyz()
        void set_euler_yxz(const Vector3 &euler)
        Vector3 get_euler_yxz()

        void set_euler(const Vector3 &euler)
        Vector3 get_euler()

        real_t dot(const Quat&)
        Quat slerp(const Quat &q, const real_t &t)
        Quat slerpni(const Quat &q, const real_t &t)
        Quat cubic_slerp(const Quat &q, const Quat &prep, const Quat &postq, const real_t &t)

        void get_axis_and_angle(Vector3 &axis, real_t &angle)
        void set_axis_angle(const Vector3 &axis, const real_t angle)

        # *=
        Quat operator*(const Quat&)
        Quat operator*(const Vector3&)

        Vector3 xform(const Vector3 &v)

        # +=
        # -=
        # *=
        # -=
        Quat operator+(const Quat&)
        Quat operator-(const Quat&)
        Quat operator-()
        Quat operator*(const real_t&)
        Quat operator/(const real_t&)

        bint operator==(const Quat&)
        bint operator!=(const Quat&)

        # String operator String()
        void set(real_t x, real_t y, real_t z, real_t w)
        Quat(real_t x, real_t y, real_t z, real_t w) except +
        Quat(const Vector3 &axis, const real_t &angle) except +
        Quat(const Vector3 &v0, const Vector3 &v1) except +
        Quat() except +

        object wrap "pythonize" ()


cdef extern from "Rect2.hpp" namespace "godot" nogil:
    ctypedef Vector2 Size2
    ctypedef Vector2 Point2

    cdef cppclass Rect2:
        Point2 position
        Size2 size

        const Vector2 &get_position()
        void set_position(const Vector2 &position)
        const Vector2 &get_size()
        void set_size(const Vector2 &size)

        real_t get_area()
        bint intersects(const Rect2 &rect)

        real_t distance_to(const Vector2 &point)
        bint intersects_transformed(const Transform2D &xform, const Rect2 &rect)
        bint intersects_segment(const Point2 &from_, const Point2 &to,
                                Point2 *position=nullptr, Point2 *normal=nullptr)
        bint encloses(const Rect2 &rect)
        bint has_no_area()
        Rect2 clip(const Rect2 &rect)
        Rect2 merge(const Rect2 &rect)
        bint has_point(const Point2 &point)
        bint no_area()

        bint operator==(const Rect2&)
        bint operator!=(const Rect2&)

        Rect2 grow(real_t by)
        Rect2 expand(Vector2 &vector)
        Rect2 expand_to(Vector2 &vector)

        # String operator String()
        Rect2() except +
        Rect2(real_t x, real_t y, real_t width, real_t height) except +
        Rect2(const Point2 &position, const Size2 &size) except +

        object wrap "pythonize" ()


cdef extern from "Ref.hpp" namespace "godot" nogil:
    cdef cppclass Ref[T]:
        # T *reference
        # void ref(const Ref&)
        # void ref_pointer(T *ref)

        bint operator<(const Ref[T]&)
        bint operator==(const Ref[T]&)
        bint operator!=(const Ref[T]&)
        T* operator*()
        # const T* operator.()
        const T *ptr()
        T *ptr()
        const T* operator*()

        # Variant operator Variant()

        Ref(const Ref[T]&) except +
        # Ref[T_Other](const Ref[T_Other]&) except +
        Ref(T*) except +
        Ref(const Variant&) except +

        void unref()
        void instance() except +
        Ref() except +


cdef extern from "RID.hpp" namespace "godot" nogil:
    cdef cppclass RID:
        # godot_rid _godot_rid
        RID() except +
        RID(__Object*) except +
        int32_t get_rid()
        bint is_valid()

        bint operator==(const RID&)
        bint operator!=(const RID&)
        bint operator<(const RID&)
        bint operator>(const RID&)
        bint operator<=(const RID&)
        bint operator>=(const RID&)

        object wrap "pythonize" ()


cdef extern from "String.hpp" namespace "godot" nogil:
    cdef cppclass CharString:
        int length()
        const char *get_data()
        object wrap "pythonize" ()

    cdef cppclass String:
        String() except +
        String(const char *) except +
        String(const wchar_t *) except +
        String(const wchar_t) except +
        String(const String&) except +
        String(object) except +

        @staticmethod
        String num(double num, int decimals=-1)

        @staticmethod
        String num_scientific(double num)

        @staticmethod
        String num_real(double num)

        @staticmethod
        String num_int64(int64_t num, int base=10, bint capitalize_hex=False)

        @staticmethod
        String chr(godot_char_type)

        @staticmethod
        String md5(const uint8_t *)

        @staticmethod
        String hex_encode_buffer(const uint8_t *buffer, int len)

        wchar_t& operator[](const int)
        wchar_t operator[](const int)

        # void operator=(const String&)
        bint operator==(const String&)
        bint operator!=(const String&)
        String operator+(const String&)
        # void operator+=(const String&)
        # void operator+=(const wchar_t)
        void operator<(const String&)
        void operator<=(const String&)
        void operator>(const String&)
        void operator>=(const String&)

        # NodePath operator NodePath()

        int length()
        const wchar_t *unicode_str()
        char *alloc_c_string()
        CharString utf8()
        CharString ascii(bint extended=False)
        str py_str()
        bytes py_bytes()

        bint begins_with(String&)
        bint begins_with_char_array(const char*)
        PoolStringArray bigrams()
        String c_escape()
        String c_unescape()
        String capitalize()
        bint empty()
        bint ends_with(String&)
        void erase(int position, int chars)
        int find(String what, int from_=0)
        int find_last(String what)
        int findn(String what, int from_=0)
        String format(Variant values)
        String format(Variant values, String placeholder)
        String get_base_dir()
        String get_basename()
        String get_extension()
        String get_file()
        int hash()
        int hex_to_int()
        String insert(int position, String what)
        bint is_abs_path()
        bint is_rel_path()
        bint is_subsequence_of(String text)
        bint is_subsequence_ofi(String text)
        bint is_valid_float()
        bint is_valid_html_color()
        bint is_valid_identifier()
        bint is_valid_integer()
        bool is_valid_ip_address()
        String json_escape()
        String left(int position)
        bint match(String expr)
        bint matchn(String expr)
        PoolByteArray md5_buffer()
        String md5_text()
        int ord_at(int)
        String pad_decimals(int digits)
        String pad_zeros(int digits)
        String percent_decode()
        String percent_encode()
        String plus_file(String file)
        String replace(String what, String forwhat)
        String replacen(String what, String forwhat)
        String rfind(String what, int from_=-1)
        String rfindn(String what, int from_=-1)
        String right(int position)
        PoolByteArray sha256_buffer()
        String sha256_text()
        float similarity(String text)
        PoolStringArray split(String divisor, bint allow_empty=True)
        PoolIntArray split_ints(String divisor, bint allow_empty=True)
        PoolRealArray split_floats(String divisor, bint allow_empty=True)
        String strip_edges(bint left=True, bint right=True)
        String substr(int from_, int len)
        float to_float()
        int64_t to_int()
        String to_lower()
        String to_upper()
        String xml_escape()
        String xml_unescape()
        signed char casecmpt_to(String)
        signed char nocasecmp_to(String)
        signed char naturalnocasecmp_to(String)
        String dedent()
        PoolStringArray rsplit(const String &divisor, const bint allow_empty=True, const int maxsplit=0)
        String rstrip(const String &chars)
        String trim_prefix(const String &prefix)
        String trim_suffix(const String &suffix)

        object wrap "pythonize" ()


cdef extern from "Transform.hpp" namespace "godot" nogil:
    cdef cppclass Transform:
        Basis basis
        Vector3 origin

        void invert()
        Transform inverse()
        void affine_invert()
        Transform affine_inverce()
        Transform rotated(const Vector3 &axis, real_t phi)
        void rotate(const Vector3 &axis, real_t phi)
        void rotate_basis(const Vector3 &axis, real_t phi)
        void set_look_at(const Vector3 &eye, const Vector3 &target, const Vector3 &up)
        Transform looking_at(const Vector3 &target, const Vector3 &up)
        void scale(const Vector3&)
        Transform scaled(const Vector3&)
        void scale_basis(const Vector3&)
        void translate(real_t tx, real_t ty, real_t tz)
        void translate(Vector3 &translation)
        Transform translated(const Vector3 &translation)

        const Basis &get_basis()
        void set_basis(const Basis &basis)
        const Vector3 &get_origin()
        void set_origin(const Vector3 &origin)

        void orthonormalize()
        Transform orthonormalized()

        bint operator==(const Transform&)
        bint operator!=(const Transform&)

        Vector3 xform(const Vector3&)
        Vector3 xform_inv(const Vector3&)

        Plane xform(const Plane&)
        Plane xform_inv(const Plane&)

        AABB xform(const AABB&)
        AABB xform_inv(const AABB&)

        # void operator*=(const Transform&)
        Transform operator*(const Transform&)
        Vector3 operator*(const Vector3&)

        Transform interpolate_with(const Transform&, real_t c)
        Transform inverse_xform(const Transform&)

        void set(real_t xx, real_t xy, real_t xz,
                 real_t yx, real_t yy, real_t yz,
                 real_t zx, real_t zy, real_t zz,
                 real_t tx, real_t ty, real_t tz)

        # String operator String()

        Transform(real_t xx, real_t xy, real_t xz,
                  real_t yx, real_t yy, real_t yz,
                  real_t zx, real_t zy, real_t zz,
                  real_t tx, real_t ty, real_t tz) except +
        Transform(const Basis &basis, const Vector3 &origin) except +
        Transform() except +

        object wrap "pythonize" ()


cdef extern from "Transform2D.hpp" namespace "godot" nogil:
    cdef cppclass Transform2D:  # C++ struct
        Vector2 elements[3]

        real_t tdotx(const Vector2&)
        real_t tdoty(const Vector2&)
        const Vector2& operator[](int)
        Vector2& operator[](int)
        Vector2 get_axis(int axis)
        void set_axis(int axis, Vector2 &vec)

        void invert()
        Transform2D inverse()
        void affine_invert()
        Transform2D affine_inverce()
        void set_rotation(real_t phi)
        real_t get_rotation()
        void set_rotation_and_scale(real_t phi, const Size2 &scale)
        void rotate(real_t phi)
        void scale(Size2 &scale)
        void scale_basis(Size2 &scale)
        void translate(real_t tx, real_t ty)
        void translate(const Vector2 &translation)

        real_t basis_determinant()
        Size2 get_scale()
        Vector2 &get_origin()
        void set_origin(const Vector2 &origin)

        Transform2D scaled(const Size2 &scale)
        Transform2D basis_scaled(const Size2 &scale)
        Transform2D translated(const Vector2 &offset)
        Transform2D rotated(real_t phi)

        Transform2D untranslated()
        void orthonormalize()
        Transform2D orthonormalized()

        bint operator==(const Transform2D&)
        bint operator!=(const Transform2D&)
        # void operator*=(const Transform2D&)
        Transform2D operator*(const Transform2D&)
        Transform2D interpolate_with(const Transform2D&, real_t c)
        Vector2 basis_xform(const Vector2 &vec)
        Vector2 basis_xform_inv(const Vector2 &vec)
        Rect2 xform(const Rect2&)
        Rect2 xform_inv(const Rect2&)

        # String operator String()

        Transform2D(real_t xx, real_t xy, real_t yx, real_t yy, real_t ox, real_t oy) except +
        Transform2D(real_t rot, Vector2 &pos) except +
        Transform2D() except +

        object wrap "pythonize" ()


cdef extern from "Variant.hpp" namespace "godot" nogil:
    cdef cppclass Variant:
        enum Type:
            NIL

            # Atomic types
            BOOL
            INT
            REAL
            STRING

            # Math types
            VECTOR2  # 5
            RECT2
            VECTOR3
            TRANSFORM2D
            PLANE
            QUAT  # 10
            RECT3
            BASIS
            TRANSFORM

            # Misc types
            COLOR
            NODE_PATH  # 15
            _RID
            OBJECT
            DICTIONARY
            ARRAY

            # Arrays
            POOL_BYTE_ARRAY  # 20
            POOL_INT_ARRAY
            POOL_REAL_ARRAY
            POOL_STRING_ARRAY
            POOL_VECTOR2_ARRAY
            POOL_VECTOR3_ARRAY  # 25
            POOL_COLOR_ARRAY

            VARIANT_MAX

        enum Operator:
            # Comparation
            OP_EQUAL
            OP_NOT_EQUAL
            OP_LESS
            OP_LESS_EQUAL
            OP_GREATER
            OP_GREATER_EQUAL

            # Mathematic
            OP_ADD
            OP_SUBSTRACT
            OP_MULTIPLY
            OP_DIVIDE
            OP_NEGATE
            OP_POSITIVE
            OP_MODULE
            OP_STRING_CONCAT

            # Bitwise
            OP_SHIFT_LEFT
            OP_SHIFT_RIGHT
            OP_BIT_AND
            OP_BIT_OR
            OP_BIT_XOR
            OP_BIT_NEGATE

            # Logic
            OP_AND
            OP_OR
            OP_XOR
            OP_NOT

            # Containment
            OP_IN
            OP_MAX

        Variant() except +
        Variant(const Variant&) except +
        Variant(bint) except +
        Variant(signed int) except +
        Variant(unsigned int) except +
        Variant(signed short) except +
        Variant(unsigned short) except +
        Variant(signed char) except +
        Variant(unsigned char) except +
        Variant(int64_t) except +
        Variant(uint64_t) except +
        Variant(float) except +
        Variant(double) except +
        Variant(const String&) except +
        Variant(const char*) except +
        Variant(const wchar_t*) except+
        Variant(const Vector2&) except +
        Variant(const Rect2&) except +
        Variant(const Vector3&) except +
        Variant(const Plane&) except +
        Variant(const AABB&) except +
        Variant(const Quat&) except +
        Variant(const Basis&) except +
        Variant(const Transform2D&) except +
        Variant(const Transform&) except +
        Variant(const Color&) except +
        Variant(const NodePath&) except +
        Variant(const RID&) except +
        Variant(const __Object&) except +
        Variant(const Dictionary&) except +
        Variant(const Array&) except +
        Variant(const PoolByteArray&) except +
        Variant(const PoolIntArray&) except +
        Variant(const PoolRealArray&) except +
        Variant(const PoolStringArray&) except +
        Variant(const PoolVector2Array&) except +
        Variant(const PoolVector3Array&) except +
        Variant(const PoolColorArray&) except +
        Variant(object) except +

        # Variant &operator=(const Variant &v)
        bool operator bool()
        # signed int operator signed int()
        # unsigned int operator unsigned int()
        # signed short operator signed short()
        # unsigned short operator unsigned short()
        # signed char operator signed char()
        # unsigned char operator unsigned char()
        # int64_t operator int64_t()
        # uint64_t operator uint64_t()

        # whar_t operator whar_t()

        # float operator float()
        # double operator double()
        # String operator String()
        # Vector2 operator Vector2()
        # Rect2 operator Rect2()
        # Vector3 operator Vector3()
        # Plane operator Plane()
        # AABB operator AABB()
        # Quat operator Quat()
        # Basis operator Basis()
        # Transform operator Transform()
        # Transform2D operator Transform2D()

        # Color operator Color()
        # NodePath operator NodePath()
        # RID operator RID()
        # godot_object* operator godot_object*()
        # T operator T*()

        # Dictionary operator Dictionary()
        # Array operator Array()

        # PoolByteArray operator PoolByteArray()
        # PoolIntArray operator PoolIntArray()
        # PoolRealArray operator PoolRealArray()
        # PoolStringArray operator PoolStringArray()
        # PoolVector2Array operator PoolVector2Array()
        # PoolVector3Array operator PoolVector3Array()
        # PoolColorArray operator PoolColorArray()

        # object operator object()

        Type get_type()
        Variant call(const String &method, const Variant **args, const int arg_count)

        bint has_method(const String &method)
        bint operator==(const Variant&)
        bint operator!=(const Variant&)
        bint operator<(const Variant&)
        bint operator<=(const Variant&)
        bint operator>(const Variant&)
        bint operator>=(const Variant&)
        bint hash_compare(const Variant&)
        bint booleanize()


cdef extern from "Vector2.hpp" namespace "godot" nogil:
    cdef cppclass Vector2:  # C++ struct
        real_t x, y, w, h # x,w and y,h are unions in C++

        Vector2(real_t, real_t)
        Vector2()

        real_t& operator[](int)
        const real_t& operator[](int)

        Vector2 operator+(const Vector2&)
        # void operator+=(const Vector2&)
        Vector2 operator-(const Vector2&)
        # void operator-=(const Vector2&)
        Vector2 operator*(const Vector2&)
        Vector2 operator*(const real_t&)
        # void operator*=(const real_t&)
        # void operator*=(const Vector2&)
        Vector2 operator/(const Vector2&)
        Vector2 operator/(const real_t&)
        # void operator/=(const real_t&)
        Vector2 operator-()

        bint operator==(const Vector2&)
        bint operator!=(const Vector2&)
        bint operator<(const Vector2&)
        bint operator<=(const Vector2&)

        void normalize()
        Vector2 normalized()
        real_t length()
        real_t length_squared()

        object wrap "pythonize" ()


cdef extern from "Vector3.hpp" namespace "godot" nogil:
    cdef cppclass Vector3:  # C++ struct
        enum Axis:
            AXIS_X
            AXIS_Y
            AXIS_Z

        real_t x, y, z

        Vector3(real_t, real_t, real_t) except +
        Vector3() except +

        const real_t& operator[](int)
        real_t operator[](int)

        # Vector3& operator+=(const Vector3&)
        Vector3 operator+(const Vector3&)
        # Vector3& operator-=(const Vector3&)
        Vector3 operator-(const Vector3&)
        # Vector3& operator*=(const Vector3&)
        Vector3 operator*(const Vector3&)
        # Vector3& operator/=(const Vector3&)
        Vector3 operator/(const Vector3&)
        # Vector3& operator*=(real_t)
        Vector3 operator*(real_t)
        # Vector3& operator/=(real_t)
        Vector3 operator/(real_t)
        Vector3 operator-()

        bint operator==(const Vector3&)
        bint operator!=(const Vector3&)
        bint operator<(const Vector3&)
        bint operator<=(const Vector3&)

        Vector3 abs()
        Vector3 ceil()
        Vector3 cross(const Vector3&)
        Vector3 linear_interpolate(const Vector3 &b, real_t t)
        Vector3 cubic_interpolate(const Vector3 &b, const Vector3 &pre_a, const Vector3 &post_b, const real_t t)
        Vector3 bounce(Vector3 &normal)
        real_t length()
        real_t length_squared()
        real_t distance_squared_to(const Vector3&)
        real_t distance_to(const Vector3&)
        real_t dot(const Vector3&)
        real_t angle_to(const Vector3&)
        Vector3 floor()
        Vector3 inverse()
        bint is_normalized()
        Basis outer(const Vector3&)
        int max_axis()
        int min_axis()
        void normalize()
        Vector3 normalized()
        Vector3 reflect()
        Vector3 rotated(const Vector3 &axis, const real_t phi)
        void rotate(const Vector3 &axis, const real_t phi)
        void snap(real_t val)
        Vector3 snapped(const float by)

        # String operator String()
        object wrap "pythonize" ()

    Vector3 vec3_cross(const Vector3 &a, Vector3 &b)


cdef extern from "Wrapped.hpp" namespace "godot" nogil:
    cdef cppclass __cpp_internal_Wrapped "godot::_Wrapped":
        godot_object *_owner
        size_t _type_tag

        _Wrapped() except +
