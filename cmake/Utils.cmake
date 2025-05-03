# ============================================================================
# Public-facing convenience functions & macros for building USD plugin(s).
# ============================================================================

# To gain access to standard install directory variables such as
# CMAKE_INSTALL_LIBDIR.
include(GNUInstallDirs)

# Exposed USD variable(s) for installation. XXX: We can hide these if we provide
# a more convenient way to install the root plugInfo.json(s) and __init__.py
# files.
set(USD_PLUGIN_DIR "plugin")
set(USD_PYTHON_DIR "python")
set(USD_PLUG_INFO_RESOURCES_DIR "resources")
set(USD_PLUG_INFO_ROOT_DIR "usd")

# Adds a USD-based C++ executable application.
function(usd_executable EXECUTABLE_NAME)

  set(options)

  set(oneValueArgs)

  set(multiValueArgs CPPFILES LIBRARIES INCLUDE_DIRS STATICFILES)

  cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  # Define a new executable.
  add_executable(${EXECUTABLE_NAME} ${args_CPPFILES})

  # Apply properties.
  _usd_target_properties(${EXECUTABLE_NAME} INCLUDE_DIRS ${args_INCLUDE_DIRS}
                         LIBRARIES ${args_LIBRARIES})

  # Add extra static files

  target_sources(${EXECUTABLE_NAME} PRIVATE ${args_STATICFILES})
  source_group("Static Files" FILES ${args_STATICFILES})

  # Install built executable.
  install(TARGETS ${EXECUTABLE_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction() # usd_executable

#
# Internal function for installing resource files (plugInfo, etc).
#
function(_usd_install_resource_files EXECUTABLE_NAME)
  set(options)

  set(oneValueArgs)

  set(multiValueArgs RESOURCE_FILES)

  cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  # Plugin resources will be installed as a 'usd' subdir under the library
  # install location.
  set(RESOURCES_INSTALL_PREFIX
      ${NAME}/resources)

  foreach(resourceFile ${args_RESOURCE_FILES})
    # Install resource file.

  endforeach()
endfunction() # _usd_install_resource_files

# Common target-specific properties to apply to library targets.
function(_usd_target_properties TARGET_NAME)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs INCLUDE_DIRS DEFINES LIBRARIES)

  cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  # Add additional platform-speific compile definitions
  set(platform_definitions)
  if(MSVC)
    # Depending on which parts of USD the project uses, additional definitions
    # for windows may need to be added. A explicit list of MSVC definitions USD
    # builds with can be found in the USD source at:
    # cmake/defaults/CXXDefaults.cmake cmake/defaults/msvcdefaults.cmake
    list(APPEND platform_definitions NOMINMAX)
  endif()

  # Some implementations of C++17 removes some deprecated functions from stl
  # MSVC adds this define by default
  if(NOT MSVC)
    list(APPEND platform_definitions BOOST_NO_CXX98_FUNCTION_BASE)
  endif()

  target_compile_definitions(${TARGET_NAME} PRIVATE ${args_DEFINES}
                                                    ${platform_definitions})

  target_compile_definitions(${TARGET_NAME}
                             PUBLIC PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

  # Exported include paths for this target.
  target_include_directories(
    ${TARGET_NAME} INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  # Project includes for building against.
  target_include_directories(
    ${TARGET_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}>)

  # Setup include path for binary dir. We set external includes as SYSTEM so
  # that their warnings are muted.
  set(_INCLUDE_DIRS "")
  list(APPEND _INCLUDE_DIRS ${args_INCLUDE_DIRS} ${USD_INCLUDE_DIR}
       ${TBB_INCLUDE_DIRS})
  if(ENABLE_PYTHON_SUPPORT)
    list(APPEND _INCLUDE_DIRS ${Python3_INCLUDE_DIR} ${Boost_INCLUDE_DIR})
  endif()
  target_include_directories(${TARGET_NAME} SYSTEM PUBLIC ${_INCLUDE_DIRS})

  # Set-up library search path.
  target_link_directories(${TARGET_NAME} PRIVATE ${USD_LIBRARY_DIR})

  # Link to libraries.
  set(_LINK_LIBRARIES "")
  set(_QT_LIBRARIES "")
  list(
    APPEND
    _QT_LIBRARIES
    Qt6::Core
    Qt6::OpenGLWidgets
    Qt6::Gui)

  list(APPEND _LINK_LIBRARIES ${args_LIBRARIES} ${TBB_LIBRARIES})
  list(
    APPEND
    _LINK_LIBRARIES
    ${Boost_PYTHON_LIBRARY}
    ${Python3_LIBRARIES}
    ${_QT_LIBRARIES})

  target_link_libraries(${TARGET_NAME} PRIVATE ${_LINK_LIBRARIES})
endfunction() # _usd_target_properties
