usd_executable(simple

    CPPFILES
        main.cpp
        camera.h
        camera.cpp
        mainwindow.h
        mainwindow.cpp
        mainwindow.ui
        mycpp/glmutils.h
        
        mycpp/mydefines.h
        mycpp/myutils.cpp
        mycpp/myutils.h
        mycpp/math_utils.h

    LIBRARIES
        sdf
        usd
        usdGeom
        tf
        gf
        usdLux
        usdAppUtils
)
