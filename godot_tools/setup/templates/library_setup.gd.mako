extends MainLoop

func _initialize():
    ProjectSettings.set("python/config/library", "${library}")
    ProjectSettings.set("python/config/module_search_path/main", "${main_zip_resource}")
    ProjectSettings.set("python/config/module_search_path/extended", "${dev_zip_resource}")

    ## % for platform, lib in python_library.items():
    ## ProjectSettings.set("python/config/python_library/${platform}", "${lib}")
    ## % endfor

    var singletons = []

    if ProjectSettings.has_setting("gdnative/singletons"):
        singletons = ProjectSettings.get("gdnative/singletons")

    var included = false
    for singleton in singletons:
        if singleton == "${library}":
            included = true
            break

    % if singleton:
    if not included:
        singletons.append("${library}")
        ProjectSettings.set("gdnative/singletons", singletons)
    % else:
    if included:
        singletons.erase("${library}")
        ProjectSettings.set("gdnative/singletons", singletons)
    % endif

    ProjectSettings.save()

func _idle(delta):
    return true
