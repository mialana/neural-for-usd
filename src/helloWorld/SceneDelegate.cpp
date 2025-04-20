#include "SceneDelegate.h"

#include "pxr/imaging/hd/camera.h"
#include "pxr/imaging/cameraUtil/conformWindow.h"
#include "pxr/imaging/pxOsd/tokens.h"

#include "pxr/imaging/hdx/renderTask.h"

#include "pxr/base/gf/range3f.h"
#include "pxr/base/gf/frustum.h"
#include "pxr/base/vt/array.h"

#include <QDebug>

SceneDelegate::SceneDelegate(pxr::HdRenderIndex *parentIndex, pxr::SdfPath const &delegateID)
 : pxr::HdSceneDelegate(parentIndex, delegateID)
{
    cameraPath = pxr::SdfPath("/camera");
    GetRenderIndex().InsertSprim(pxr::HdPrimTypeTokens->camera, this, cameraPath);
    pxr::GfFrustum frustum;
    frustum.SetPosition(pxr::GfVec3d(0, 0, 3));
    SetCamera(frustum.ComputeViewMatrix(), frustum.ComputeProjectionMatrix());

    GetRenderIndex().InsertRprim(pxr::HdPrimTypeTokens->mesh, this, pxr::SdfPath("/triangle") );

}

void
SceneDelegate::AddRenderTask(pxr::SdfPath const &id)
{
    GetRenderIndex().InsertTask<pxr::HdxRenderTask>(this, id);
    _ValueCache &cache = _valueCacheMap[id];
    cache[pxr::HdTokens->collection] = pxr::HdRprimCollection(pxr::HdTokens->geometry, pxr::HdReprSelector(pxr::HdReprTokens->smoothHull));
}

void
SceneDelegate::AddRenderSetupTask(pxr::SdfPath const &id)
{
    GetRenderIndex().InsertTask<pxr::HdxRenderSetupTask>(this, id);
    _ValueCache &cache = _valueCacheMap[id];
    pxr::HdxRenderTaskParams params;
    params.camera = cameraPath;
    params.viewport = pxr::GfVec4f(0, 0, 512, 512);
    cache[pxr::HdTokens->params] = pxr::VtValue(params);
}

void SceneDelegate::SetCamera(pxr::GfMatrix4d const &viewMatrix, pxr::GfMatrix4d const &projMatrix)
{
    SetCamera(cameraPath, viewMatrix, projMatrix);
}

void SceneDelegate::SetCamera(pxr::SdfPath const &cameraId, pxr::GfMatrix4d const &viewMatrix, pxr::GfMatrix4d const &projMatrix)
{
    _ValueCache &cache = _valueCacheMap[cameraId];
    cache[pxr::HdCameraTokens->windowPolicy] = pxr::VtValue(pxr::CameraUtilFit);

    GetRenderIndex().GetChangeTracker().MarkSprimDirty(cameraId, pxr::HdCamera::AllDirty);
}


pxr::VtValue SceneDelegate::Get(pxr::SdfPath const &id, const pxr::TfToken &key)
{
    qDebug() << "[" << id.GetString() <<"][" << key.GetString() << "]";
    _ValueCache *vcache = pxr::TfMapLookupPtr(_valueCacheMap, id);
    pxr::VtValue ret;
    if (vcache && pxr::TfMapLookup(*vcache, key, &ret)) {
        return ret;
    }

    if (key == pxr::HdShaderTokens->fragmentShader)
    {
        return pxr::VtValue();
    }

    if (key == pxr::HdTokens->points)
    {
        pxr::VtVec3fArray points;

        points.push_back(pxr::GfVec3f(0,0,0));
        points.push_back(pxr::GfVec3f(1,0,0));
        points.push_back(pxr::GfVec3f(0,1,0));
        return pxr::VtValue(points);
    }
}

bool SceneDelegate::GetVisible(pxr::SdfPath const &id)
{
    qDebug() << "[" << id.GetString() <<"][Visible]";
    return true;
}

pxr::GfRange3d SceneDelegate::GetExtent(pxr::SdfPath const &id)
{
    qDebug() << "[" << id.GetString() <<"][Extent]";
    return pxr::GfRange3d(pxr::GfVec3d(-1,-1,-1), pxr::GfVec3d(1,1,1));
}

pxr::GfMatrix4d SceneDelegate::GetTransform(pxr::SdfPath const &id)
{
    qDebug() << "[" << id.GetString() <<"][Transform]";
    return pxr::GfMatrix4d(1.0f);
}

pxr::HdMeshTopology SceneDelegate::GetMeshTopology(pxr::SdfPath const &id)
{
    qDebug() << "[" << id.GetString() <<"][Topology]";
    pxr::VtArray<int> vertCountsPerFace;
    pxr::VtArray<int> verts;
    vertCountsPerFace.push_back(3);
    verts.push_back(0);
    verts.push_back(1);
    verts.push_back(2);

    pxr::HdMeshTopology triangleTopology(pxr::PxOsdOpenSubdivTokens->none, pxr::HdTokens->rightHanded, vertCountsPerFace, verts);
    return triangleTopology;
}

pxr::HdPrimvarDescriptorVector SceneDelegate::GetPrimvarDescriptors(pxr::SdfPath const& id, pxr::HdInterpolation interpolation)
{
    qDebug() << "[" << id.GetString() <<"][GetPrimvarDescriptors]";
    pxr::HdPrimvarDescriptorVector primvarDescriptors;

    if (interpolation == pxr::HdInterpolation::HdInterpolationVertex)
    {
        primvarDescriptors.push_back(pxr::HdPrimvarDescriptor(pxr::HdTokens->points, interpolation));
    }


    return primvarDescriptors;
}
