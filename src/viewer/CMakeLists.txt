file(GLOB_RECURSE SOURCEFILES *.cpp *.h *.ui)
file(GLOB STATICFILES *.usd *.usd* *.txt)
list(FILTER STATICFILES EXCLUDE REGEX "CMakeLists.txt")

usd_executable(viewer

    CPPFILES
        ${SOURCEFILES}
        
    LIBRARIES
        usd
        usdGeom
        usdLux
        usdImaging
        usdAppUtils
        gf
        glf
        hd
        hdx
        sdf
        tf
    
    STATICFILES
        ${STATICFILES}
)
