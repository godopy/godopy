def using_clang(env):
    return False

def disable_warnings(self):
    # 'self' is the environment
    if self.msvc and not using_clang(self):
        # We have to remove existing warning level defines before appending /w,
        # otherwise we get: "warning D9025 : overriding '/W3' with '/w'"
        self["CCFLAGS"] = [x for x in self["CCFLAGS"] if not (x.startswith("/W") or x.startswith("/w"))]
        self["CFLAGS"] = [x for x in self["CFLAGS"] if not (x.startswith("/W") or x.startswith("/w"))]
        self["CXXFLAGS"] = [x for x in self["CXXFLAGS"] if not (x.startswith("/W") or x.startswith("/w"))]
        self.AppendUnique(CCFLAGS=["/w"])
    else:
        self.AppendUnique(CCFLAGS=["-w"])
