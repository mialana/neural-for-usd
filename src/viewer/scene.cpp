#include "scene.h"

#include <QDebug>

#include <pxr/base/gf/frustum.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usdImaging/usdImaging/sceneIndices.h>

Scene::Scene()
    : m_stage(nullptr)
    , m_finalSceneIndex(nullptr)
{}

void Scene::initialize(std::string& stagePath, SdfPath cameraPath)
{
    m_stage = UsdStage::Open(stagePath);
    if (!TF_VERIFY(m_stage)) {
        qWarning() << "Failed to open stage";
        return;
    }

    m_primaryCamera = UsdGeomCamera::Define(m_stage, cameraPath);


    UsdImagingCreateSceneIndicesInfo info;
    info.displayUnloadedPrimsWithBounds = false;

    const UsdImagingSceneIndices sceneIndices = UsdImagingCreateSceneIndices(info);
    m_finalSceneIndex = sceneIndices.stageSceneIndex;
    if (!TF_VERIFY(m_finalSceneIndex)) {
        qWarning() << "Failed to get stage scene index";
        return;
    }

    m_finalSceneIndex->SetStage(m_stage);
    m_finalSceneIndex->SetTime(UsdTimeCode::Default());

    // for (const UsdPrim& p : m_stage->Traverse()) {
    //     HdSceneIndexPrim ip = m_sceneIndex->GetPrim(p.GetPrimPath());
    //     qDebug() << ip.primType.GetString();
    //     HdDebugPrintDataSource(std::cout, ip.dataSource, 1);
    // }
}

UsdImagingStageSceneIndexRefPtr Scene::getFinalSceneIndex() const {
    return m_finalSceneIndex;
}

GfMatrix4d Scene::getCamView()
{
    return m_primaryCamera.GetCamera(0).GetFrustum().ComputeViewMatrix();
}
GfMatrix4d Scene::getCamProj()
{
    return m_primaryCamera.GetCamera(0).GetFrustum().ComputeProjectionMatrix();
};
