usd_executable(simple

    CPPFILES
        main.cpp
        camera.h
        camera.cpp
        glm_utils.h
        math_utils.h
        mainwindow.h
        mainwindow.cpp
        mainwindow.ui
        mycpp/glmutils.h
        mycpp/MyCpp_global.h
        mycpp/mydefines.h
        mycpp/myutils.cpp
        mycpp/myutils.h

    LIBRARIES
        sdf
        usd
        usdGeom
        tf
        gf
        usdLux
)
