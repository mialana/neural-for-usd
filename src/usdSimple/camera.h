#pragma once

// #define GLM_FORCE_RADIANS
// // Primary GLM library
#include <glm/glm.hpp>
// // For glm::translate, glm::rotate, and glm::scale.
#include <glm/gtc/matrix_transform.hpp>
// // For glm::to_string.
// #include <glm/gtx/string_cast.hpp>
// // For glm::value_ptr.
// #include <glm/gtc/type_ptr.hpp>

#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/base/gf/camera.h>
#include <vector>

class Camera
{
private:
    pxr::UsdGeomCamera m_usdCamera;
    pxr::GfCamera m_usdCameraParams;

    void createUsdCameraParams();
    void createUsdCamera(const pxr::UsdStagePtr& stage, const char* path);

public:
    Camera();
};
