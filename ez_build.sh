#!/usr/bin/env bash

cmake_install_prefix="/Users/Dev/Projects/Neural-for-USD"

find src/usdSimple -iname '*.h' -o -iname '*.cpp' | xargs clang-format -i

if [ -f japanesePlaneToy.usda ]; then
  trash japanesePlaneToy.usda
fi

if [ -e build ]; then
  trash build
  mkdir build
fi

cd build
cmake \
  -DUSD_ROOT="/opt/usd" \
  -DCMAKE_INSTALL_PREFIX=${cmake_install_prefix} \
  ..

cmake --build . -j8 -- simple