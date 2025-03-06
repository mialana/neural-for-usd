file(GLOB ROOT_HEADER *.h)

add_library(MyCpp
    ${ROOT_HEADER}
)

target_include_directories(glm-header-only INTERFACE
	"$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
	"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)