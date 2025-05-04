#include "freecamera.h"

#include "utils/gfaddins.h"

#include <mycpp/myglm.h>

#include <pxr/base/gf/rotation.h>

#include <QDebug>

FreeCamera::FreeCamera(float width, float height, float aspectRatio, float fov, GfFrustum frustum)
    : aspectRatio(aspectRatio)
    , fieldOfView(fov)
    , gfFrustum(frustum)
{
    recomputeAttributes();
}

FreeCamera::FreeCamera(float width, float height, GfFrustum frustum)
    : FreeCamera(width, height, width / (float)height, frustum.GetFOV(), frustum)
{}

FreeCamera::FreeCamera(float width, float height, const FreeCamera& other)
    : FreeCamera(width, height, other.aspectRatio, other.fieldOfView, other.gfFrustum)
{}

void FreeCamera::setFromGfCamera(const GfCamera& gfCamera)
{
    gfFrustum = gfCamera.GetFrustum();

    // update FOV, maintain aspect ratio
    fieldOfView = gfCamera.GetFieldOfView(GfCamera::FOVHorizontal);

    GfRange1f range = gfCamera.GetClippingRange();

    gfFrustum.SetPerspective(fieldOfView, aspectRatio, range.GetMin(), range.GetMax());

    recomputeAttributes();
    return;
}

GfCamera FreeCamera::createGfCamera()
{
    GfCamera c = GfCamera();

    GfMatrix4d viewMatrix = gfFrustum.ComputeViewMatrix();

    GfMatrix4d projMatrix = gfFrustum.ComputeProjectionMatrix();

    c.SetFromViewAndProjectionMatrix(viewMatrix, projMatrix);

    return c;
}

void FreeCamera::translateAlongRight(float amt)
{
    recomputeAttributes();
    gfFrustum.SetPosition(eye + right * amt);
}

void FreeCamera::translateAlongUp(float amt)
{
    recomputeAttributes();
    gfFrustum.SetPosition(eye + up * amt);
}

void FreeCamera::rotateAboutUp(float deg)
{
    recomputeAttributes();

    GfVec3d newRef = ref - eye;
    newRef = GfMatrix4d().SetRotate(GfRotation(GfVec3d(up), deg)).Transform(newRef);
    newRef = newRef + eye;

    GfMatrix4d camToWorld = GfMatrix4d().SetLookAt(eye, newRef, up).GetInverse();

    gfFrustum.SetPositionAndRotationFromMatrix(camToWorld);
}

void FreeCamera::rotateAboutRight(float deg)
{
    recomputeAttributes();

    GfVec3d newRef = ref - eye;
    newRef = GfMatrix4d().SetRotate(GfRotation(GfVec3d(right), deg)).Transform(newRef);
    newRef = newRef + eye;

    GfMatrix4d camToWorld = GfMatrix4d().SetLookAt(eye, newRef, up).GetInverse();

    gfFrustum.SetPositionAndRotationFromMatrix(camToWorld);
}

void FreeCamera::rotateTheta(float deg)
{
    recomputeAttributes();

    GfMatrix4d rot = GfMatrix4d().SetRotate(GfRotation(right, deg));

    gfFrustum.Transform(rot);
}

void FreeCamera::rotatePhi(float deg)
{
    recomputeAttributes();
    GfMatrix4d rot = GfMatrix4d().SetRotate(GfRotation(up, deg));

    gfFrustum.Transform(rot);
}

void FreeCamera::orbitAboutOrigin(float theta, float phi)
{

    recomputeAttributes();
    GfVec3d target = GfVec3d(0.0);
    GfVec3d newEye = GfMatrix4d().SetRotate(GfRotation(right, theta)).Transform(eye);
    newEye = GfMatrix4d().SetRotate(GfRotation(up, phi)).Transform(newEye);

    GfMatrix4d camToWorld = GfMatrix4d().SetLookAt(newEye, target, up).GetInverse();

    gfFrustum.SetPositionAndRotationFromMatrix(camToWorld);
}

void FreeCamera::zoom(float amt)
{
    recomputeAttributes();
    gfFrustum.SetPosition(eye + forward * amt);
}

float FreeCamera::getRadius()
{
    recomputeAttributes();
    return radius;
}

float FreeCamera::getTheta()
{
    recomputeAttributes();
    return theta;
}

float FreeCamera::getPhi()
{
    recomputeAttributes();
    return phi;
}

void FreeCamera::recomputeAttributes()
{
    gfFrustum.ComputeViewFrame(&right, &up, &forward);

    eye = gfFrustum.GetPosition();
    ref = eye + forward;

    // Compute spherical attributes based on target ref
    const GfVec3d targetRef = GfVec3d(0.0);
    GfVec3d ray = eye - targetRef;
    radius = ray.GetLength();
    theta = std::acos(ray[1] / radius);
    phi = std::atan2(ray[2], ray[0]);
}
