cdef extern from "godot_cpp/variant/color.hpp" namespace "godot" nogil:
    cdef cppclass Color:
        real_t r
        real_t g
        real_t b
        real_t a

        real_t[4] components

        Color()
        Color(float, float, float, float)
        Color(float, float, float)
        Color(const Color &, float)
        Color(const String &)
        Color(str)
        Color(const String &, float)
        Color(str, float)

        uint32_t to_rgba32() const
        uint32_t to_argb32() const
        uint32_t to_abgr32() const
        uint64_t to_rgba64() const
        uint64_t to_argb64() const
        uint64_t to_abgr64() const

        str to_html() const
        str to_html(bint) const
        float get_h() const
        float get_s() const
        float get_v() const
        void set_hsv(float p_h, float p_s, float p_v)
        void set_hsv(float p_h, float p_s, float p_v, float p_alpha)

        float &operator[](int)
        const float &operator[](int)
        bint operator==(const Color &)
        bint operator!=(const Color &)
        Color operator-() const
        Color operator-(const Color &) const
        Color operator*(const Color &) const
        Color operator*(float) const
        Color operator/(const Color &) const
        Color operator/(float) const

        bint is_equal_approx(const Color &p_color) const
        Color clamp() const
        Color clamp(const Color &p_min) const
        Color clamp(const Color &p_min, const Color &p_max) const
        void invert()
        Color inverted() const

        float get_luminance() const
        Color lerp(const Color &p_to, float p_weight)
        Color darkened(float p_amount) const
        Color lightened(float p_amount)

        uint32_t to_rgbe9995()

        Color blend(const Color &p_over) const
        Color srgb_to_linear() const
        Color linear_to_srgb() const

        @staticmethod
        Color hex(uint32_t p_hex)

        @staticmethod
        Color hex64(uint64_t p_hex)

        @staticmethod
        Color html(str)

        @staticmethod
        bint html_is_valid(str)

        @staticmethod
        Color named(str)

        @staticmethod
        Color named(str, const Color &p_default)

        @staticmethod
        int find_named_color(str)

        @staticmethod
        int get_named_color_count()

        @staticmethod
        str get_named_color_name(int p_idx)

        @staticmethod
        Color get_named_color(int p_idx)

        @staticmethod
        Color from_string(str, str)

        @staticmethod
        Color from_hsv(float p_h, float p_s, float p_v)

        @staticmethod
        Color from_hsv(float p_h, float p_s, float p_v, float p_alpha)

        @staticmethod
        Color from_rgbe9995(uint32_t p_rgbe)
