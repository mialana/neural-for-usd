#include "mycamera.h"
#include <pxr/base/gf/rotation.h>
#include <mycpp/myglm.h>
#include <QDebug>

MyCamera::MyCamera(float width, float height)
    : MyCamera(width, height, GfVec3d(0.0, 0.0, 25.0), GfVec3d(0.0, 0.0, 0.0), GfVec3d(0.0, 1.0, 0.0))
{}

MyCamera::MyCamera(float width, float height, GfVec3d eye, GfVec3d ref, GfVec3d worldUp)
    : eye(eye)
    , ref(ref)
    , worldUp(worldUp)
    , forward(GfVec3d(0.0, 0.0, -1.0))
    , right(GfVec3d(1.0, 0.0, 0.0))
{
    aspectRatio = width / (float)height;
    fieldOfView = 30;

    recomputeAttributes();
}

GfCamera MyCamera::createGfCamera()
{
    GfCamera c = GfCamera();

    GfMatrix4d viewMatrix = GfMatrix4d().SetLookAt(eye, ref, worldUp);
    GfMatrix4d projMatrix = this->createProjectionMatrix();

    c.SetFromViewAndProjectionMatrix(viewMatrix, projMatrix);

    return c;
}

void MyCamera::translateAlongForward(float amt)
{
    GfVec3d translation = forward * amt;
    eye += translation;
    ref += translation;
}

void MyCamera::translateAlongRight(float amt)
{
    GfVec3d translation = right * amt;
    eye += translation;
    ref += translation;
}

void MyCamera::translateAlongUp(float amt)
{
    GfVec3d translation = worldUp * amt;
    eye += translation;
    ref += translation;
}

void MyCamera::rotateAboutUp(float deg)
{
    GfMatrix4d rot = GfMatrix4d(1.0).SetRotate(GfRotation(worldUp, deg));
    ref = ref - eye;
    GfVec4d ref4d = rot * GfVec4d(ref[0], ref[1], ref[2], 1.f);
    ref = GfVec3d(ref4d[0], ref4d[1], ref4d[2]);
    ref = ref + eye;
    recomputeAttributes();
}

void MyCamera::rotateAboutRight(float deg)
{
    GfMatrix4d rot = GfMatrix4d(1.0).SetRotate(GfRotation(right, deg));
    ref = ref - eye;
    GfVec4d ref4d = rot * GfVec4d(ref[0], ref[1], ref[2], 1.f);
    ref = GfVec3d(ref4d[0], ref4d[1], ref4d[2]);
    ref = ref + eye;
    recomputeAttributes();
}

void MyCamera::rotateTheta(float deg)
{
    GfMatrix4d rot = GfMatrix4d().SetIdentity().SetRotateOnly(GfRotation(right, deg));
    eye = eye - ref;
    GfVec4d eye4d = rot * GfVec4d(eye[0], eye[1], eye[2], 1.f);
    eye = GfVec3d(eye4d[0], eye4d[1], eye4d[2]);
    eye = eye + ref;
    recomputeAttributes();
}

void MyCamera::rotatePhi(float deg)
{
    GfMatrix4d rot = GfMatrix4d(1.0).SetIdentity().SetRotateOnly(GfRotation(worldUp, deg));
    eye = eye - ref;
    GfVec4d eye4d = rot * GfVec4d(eye[0], eye[1], eye[2], 1.f);
    eye = GfVec3d(eye4d[0], eye4d[1], eye4d[2]);
    eye = eye + ref;
    recomputeAttributes();
}

void MyCamera::zoom(float amt)
{
    GfVec3d translation = forward * amt;
    eye += translation;
}

void MyCamera::recomputeAttributes()
{
    forward = (ref - eye).GetNormalized();
    right = GfCross(forward, worldUp).GetNormalized();
    worldUp = GfCross(right, forward);
}

GfMatrix4d MyCamera::createProjectionMatrix()
{
    float near = 0.01;
    float far = 100.f;

    glm::mat4 p = glm::perspective(glm::radians(fieldOfView), aspectRatio, near, far);

    GfMatrix4d m = GfMatrix4d(p[0][0],
                              p[0][1],
                              p[0][2],
                              p[0][3],
                              p[1][0],
                              p[1][1],
                              p[1][2],
                              p[1][3],
                              p[2][0],
                              p[2][1],
                              p[2][2],
                              p[2][3],
                              p[3][0],
                              p[3][1],
                              p[3][2],
                              p[3][3]);
    return m;
}
