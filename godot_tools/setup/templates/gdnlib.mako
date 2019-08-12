[general]

singleton=${'true' if singleton else 'false'}
load_once=${'true' if load_once else 'false'}
symbol_prefix="${symbol_prefix}"
reloadable=${'true' if reloadable else 'false'}

[entry]

% for platform, lib in libraries.items():
${platform}="res://${lib}"
% endfor

[dependencies]

% for platform, deps in dependencies.items():
${platform}=[ ${', '.join('"%s"' % d for d in deps)} ]
% endfor
