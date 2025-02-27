#pragma once

#define GLM_FORCE_SWIZZLE
#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

/* custom typedefs and define macros */
typedef glm::vec4 Point4f;  // w = 1, theoretically
typedef glm::vec3 Point3f;
typedef glm::ivec3 Point3i;
typedef glm::vec2 Point2f;
typedef glm::ivec2 Point2i;

typedef glm::vec4 Vector4f;  // w = 0, theoretically
typedef glm::vec3 Vector3f;
typedef glm::vec2 Vector2f;
typedef glm::ivec2 Vector2i;

typedef glm::vec3 Normal3f;

typedef glm::vec3 Color3f;   // should always be in range 0-1
typedef glm::ivec3 Color3i;  // should always be in range 0-255

typedef float AngleRad;      // should always be in radians

typedef glm::vec2 PixelCoord;
typedef glm::vec2 NDCoord;  // Normalized Device Coordinates

typedef glm::vec3 qfABC;    // glm::vec3 contains components of quadratic formula.
