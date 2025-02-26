usd_executable(simple

    CPPFILES
        main.cpp
        camera.h
        camera.cpp

    LIBRARIES
        sdf
        usd
        usdGeom
        tf
        gf
)