file(GLOB SOURCEFILES *.cpp *.h *.ui)
file(GLOB STATICFILES *.usd *.usd* *.txt)
list(FILTER STATICFILES EXCLUDE REGEX "CMakeLists.txt")

usd_executable(base

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
        sdf
        tf
    
    STATICFILES
        ${STATICFILES}
)
