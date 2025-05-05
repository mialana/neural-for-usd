set(PYTHON_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/main.py")

add_custom_target(run_nerf_python
    COMMAND ${CMAKE_COMMAND} -E echo "Running Python tool..."
    COMMAND /Users/Dev/.pyenv/versions/neural/bin/python ${CMAKE_BINARY_DIR}/main.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running main.py"
)