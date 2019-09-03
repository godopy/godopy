[plugin]

name="${name}"
description="${description or name}"
% if author:
author="${author}"
% endif
version="${version or '0.0.0'}"
script="${script}"
