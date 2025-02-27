usd_executable(simple

    CPPFILES
        main.cpp
        camera.h
        camera.cpp
        glm_utils.h
        math_utils.h

    LIBRARIES
        sdf
        usd
        usdGeom
        tf
        gf
)