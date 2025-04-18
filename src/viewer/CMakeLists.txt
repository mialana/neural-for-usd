file(GLOB CPPFILES *.cpp)
file(GLOB HEADERFILES *.h)
file(GLOB UIFILES *.ui)
file(GLOB RESOURCE_FILES *.usd *.usd* *.txt)

usd_executable(viewer

    CPPFILES
        ${CPPFILES}
        ${HEADERFILES}
        ${UIFILES}
        
    LIBRARIES
        sdf
        usd
        usdGeom
        tf
        gf
        usdLux
        usdAppUtils
        usdUtils
        usdviewq
        usdImaging
        glf

    RESOURCE_FILES
        ${RESOURCE_FILES}
)
