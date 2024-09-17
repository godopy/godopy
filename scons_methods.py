import os

def normalize_path(val, env):
    return val if os.path.isabs(val) else os.path.join(env.Dir('#').abspath, val)


def validate_parent_dir(key, val, env):
    if not os.path.isdir(normalize_path(os.path.dirname(val), env)):
        raise UserError("'%s' is not a directory: %s" % (key, os.path.dirname(val)))

def disable_warnings(env):
    if env['platform'] == 'windows':
        # We have to remove existing warning level defines before appending /w,
        # otherwise we get: "warning D9025 : overriding '/W3' with '/w'"
        env["CCFLAGS"] = [x for x in env["CCFLAGS"] if not (x.startswith("/W") or x.startswith("/w"))]
        env["CFLAGS"] = [x for x in env["CFLAGS"] if not (x.startswith("/W") or x.startswith("/w"))]
        env["CXXFLAGS"] = [x for x in env["CXXFLAGS"] if not (x.startswith("/W") or x.startswith("/w"))]
        env.AppendUnique(CCFLAGS=["/w"])
    else:
        env.AppendUnique(CCFLAGS=["-w"])
