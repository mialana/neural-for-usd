#!/usr/bin/env bash

cmake_install_prefix="/Users/liu.amy05/Documents/Neural-for-USD"

trash build
mkdir build
cd build
cmake \
  -DUSD_ROOT="/Users/liu.amy05/usd" \
  -DCMAKE_INSTALL_PREFIX=${cmake_install_prefix} \
  ..

cmake --build . -j8 -- simple