#pragma once

#include <pxr/base/gf/frustum.h>
#include <pxr/base/gf/camera.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/matrix4d.h>

PXR_NAMESPACE_USING_DIRECTIVE

class FreeCamera
{
private:
    GfFrustum gfFrustum;

    float aspectRatio;
    float fieldOfView;

    GfVec3d eye;
    GfVec3d ref;

    GfVec3d forward;
    GfVec3d right;
    GfVec3d up;

    float theta;
    float phi;
    float radius;

    void recomputeAttributes();

public:
    FreeCamera(float width, float height, float aspectRatio, float fov, GfFrustum frustum);
    FreeCamera(float width, float height, GfFrustum frustum);
    FreeCamera(float width, float height, const FreeCamera& other);

    void setFromGfCamera(const GfCamera& gfCamera);
    GfCamera createGfCamera();

    void translateAlongRight(float amt);
    void translateAlongUp(float amt);

    void rotateAboutUp(float deg);
    void rotateAboutRight(float deg);

    void rotateTheta(float deg);
    void rotatePhi(float deg);

    void orbitAboutOrigin(float theta, float phi);

    void zoom(float amt);

    float getRadius();
    float getTheta();
    float getPhi();

};
