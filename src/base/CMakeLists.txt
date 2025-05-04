file(GLOB SOURCEFILES *.cpp *.h *.ui)
file(GLOB STATICFILES *.usd *.usd* *.txt, *.qrc)
list(FILTER STATICFILES EXCLUDE REGEX "CMakeLists.txt")

usd_executable(base

    CPPFILES
        ${SOURCEFILES}
        utils/gfsampling.h
        
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
