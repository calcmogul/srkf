#!/bin/bash
find . -type f \( -name '*.cpp' -o -name '*.hpp' \) -not -path './build*/*' -exec clang-format -i {} \;
find . -type f \( -name CMakeLists.txt -o -name '*.cmake' \) -not -path './build*/*' -exec python -m gersemi -i --no-warn-about-unknown-commands {} \;
