#pragma once

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usdImaging/usdImaging/stageSceneIndex.h>

PXR_NAMESPACE_USING_DIRECTIVE

class Scene
{
public:
    Scene();

    void initialize(std::string& stagePath, SdfPath cameraPath);

    UsdImagingStageSceneIndexRefPtr getFinalSceneIndex() const;

    GfMatrix4d getCamView();
    GfMatrix4d getCamProj();
private:
    UsdStageRefPtr m_stage;
    UsdGeomCamera m_primaryCamera;
    UsdImagingStageSceneIndexRefPtr m_finalSceneIndex;
};
