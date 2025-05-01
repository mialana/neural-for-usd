#pragma once

#include <pxr/usd/usdGeom/camera.h>
#include <pxr/base/gf/camera.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/matrix4d.h>



PXR_NAMESPACE_USING_DIRECTIVE

class MyCamera
{
private:
    float aspectRatio;
    float fieldOfView; // FOVY

    GfVec3d forward;
    GfVec3d right;

    float theta;
    float phi;
    float radius;

    GfMatrix4d createProjectionMatrix();
public:
    GfVec3d eye;
    GfVec3d ref;
    GfVec3d worldUp;

    MyCamera(float width, float height);
    MyCamera(float width, float height, GfVec3d eye, GfVec3d ref, GfVec3d worldUp);

    GfCamera createGfCamera();

    void recomputeAttributes();

    void translateAlongForward(float amt);
    void translateAlongRight(float amt);
    void translateAlongUp(float amt);

    void rotateAboutUp(float deg);
    void rotateAboutRight(float deg);

    void rotateTheta(float deg);
    void rotatePhi(float deg);

    void zoom(float amt);
};
