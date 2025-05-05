file(GLOB SOURCEFILES *.cpp)
file(GLOB STATICFILES *.py)
list(FILTER STATICFILES EXCLUDE REGEX "CMakeLists.txt")

usd_executable(nerf
    CPPFILES
        ${SOURCEFILES}

    STATICFILES
        ${STATICFILES}
)
