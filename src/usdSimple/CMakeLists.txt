usd_executable(simple

    CPPFILES
        main.cpp
        camera.h
        camera.cpp
        mainwindow.h
        mainwindow.cpp
        camerapose.h
        camerapose.cpp
        scene.h
        scene.cpp
        mainwindow.ui
        
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
)
