add_subdirectory(usdSimple)

# Install top level python __init__.py
install(FILES __init__.py
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${USD_PYTHON_DIR}/${ORGANIZATION})

# Mirror installation structure in intermediate binary dir for running tests
# against.
if(BUILD_TESTING)
  file(
    COPY __init__.py
    DESTINATION
      ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${USD_PYTHON_DIR}/${ORGANIZATION}
  )
endif()
