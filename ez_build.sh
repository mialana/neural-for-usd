#!/usr/bin/env bash

cmake_install_prefix="/Users/Dev/Projects/Neural-for-USD"

# if [ -e build ]; then
#   trash build
#   mkdir build
# fi

cd build
cmake \
  -DUSD_ROOT="/opt/usd" \
  -DCMAKE_INSTALL_PREFIX=${cmake_install_prefix} \
  ..

cmake --build . -j8 -- clean base