# Default build configurations for the USDPluginExamples project.

set(USD_ROOT /Users/liu.amy05/usd)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
# Find Qt
set(CMAKE_PREFIX_PATH /Users/liu.amy05/Qt/6.8.1/macos)
set(CMAKE_BUILD_TYPE "Release")

# To find Qt UI files
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_OSX_ARCHITECTURES "x86_64")

# By default, build for release.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Check if CTest should be enabled.
if (BUILD_TESTING)
    enable_testing()

    # Be very verbose on test failure.
    list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
endif()

if (MSVC)
    # From OpenUSD/cmake/defaults/msvcdefaults.cmake
    #
    # The /Zc:inline option strips out the "arch_ctor_<name>" symbols used for
    # library initialization by ARCH_CONSTRUCTOR starting in Visual Studio 2019,
    # causing release builds to fail. Disable the option for this and later
    # versions.
    #
    # For more details, see:
    # https://developercommunity.visualstudio.com/content/problem/914943/zcinline-removes-extern-symbols-inside-anonymous-n.html
    if (MSVC_VERSION GREATER_EQUAL 1920)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:inline-")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:inline")
    endif()
endif()
