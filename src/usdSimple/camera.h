#pragma once

#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/base/gf/camera.h>

#define GLMMat4ToGF(_glmMtxPtr) (*reinterpret_cast<pxr::GfMatrix4d*>(_glmMtxPtr))

class Camera
{
private:
    pxr::UsdGeomXform m_usdCameraXform;
    pxr::UsdGeomCamera m_usdCamera;
    pxr::GfCamera m_usdCameraParams;

    void createUsdCameraParams();
    void createUsdCamera(const pxr::UsdStagePtr& stage, const char* path);
    void generateCameraTransforms(const pxr::UsdStagePtr& stage, int numSamples);

public:
    Camera();
};
