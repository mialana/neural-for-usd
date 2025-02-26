#include "camera.h"

#include <iostream>

#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/usdImaging/usdviewq/utils.h>



void Camera::createUsdCameraParams()
{
    m_usdCameraParams = pxr::GfCamera();
    pxr::GfMatrix4d m = pxr::GfMatrix4d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    m_usdCameraParams.SetTransform(m);

    return;
}

void Camera::createUsdCamera(const pxr::UsdStagePtr& stage, const char* path)
{
    const pxr::SdfPath& cameraPath = pxr::SdfPath(path);
    m_usdCamera = pxr::UsdGeomCamera::Define(stage, cameraPath);
    m_usdCamera.CreateProjectionAttr().Set(pxr::UsdGeomTokens->perspective);

    createUsdCameraParams();

    return;
}

void setInitial()
{
    std::vector<glm::vec3> bbox
        = {glm::vec3(-0.4730462431907654, -0.5500000016279251, -8.960636894995694e-8),
           glm::vec3(0.4730462431907654, 0.5499999523162841, 0.30267277379902635)};

    glm::mat4 xform = glm::mat4(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);

    return;
}

Camera::Camera()
{
    // Path to the USD file
    std::string filePath
        = "/Users/liu.amy05/Documents/Neural-for-USD/assets/japanesePlaneToy/japanesePlaneToy.usda";

    std::cout << filePath << std::endl;

    // Create a USD stage
    pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(filePath);

    if (!stage) {
        std::cout << "Failed to open USD stage." << std::endl;
        return;
    }

    createUsdCamera(stage, "/MyCam");

    stage->Export("/Users/liu.amy05/Documents/Neural-for-USD/japanesePlaneToy.usda");

    std::cout << "Done" << std::endl;

    return;
}
