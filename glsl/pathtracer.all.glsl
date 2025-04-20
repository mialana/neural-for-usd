#version 330 core

// Uniforms
uniform vec3 u_Eye;                  // Camera position
uniform vec3 u_Forward;              // Camera forward vector
uniform vec3 u_Right;                // Camera right vector
uniform vec3 u_Up;                   // Camera up vector

uniform int u_Iterations;            // How many rays per pixel have been cast
uniform vec2 u_ScreenDims;           // Screen width and height

uniform sampler2D u_AccumImg;        // A texture storing the accumulation of
                                     // all previous iterations' color values
uniform sampler2D u_EnvironmentMap;  // An HDR image of an environment map

// Varyings
in vec3 fs_Pos;
in vec2 fs_UV;
out vec4 out_Col;

// Numeric constants
#define PI 3.14159265358979323
#define TWO_PI 6.28318530717958648
#define FOUR_PI 12.5663706143591729
#define INV_PI 0.31830988618379067
#define INV_TWO_PI 0.15915494309
#define INV_FOUR_PI 0.07957747154594767
#define PI_OVER_TWO 1.57079632679489662
#define ONE_THIRD 0.33333333333333333
#define E 2.71828182845904524
#define INFINITY 1000000.0
#define OneMinusEpsilon 0.99999994
#define RayEpsilon 0.000005

// Path tracer recursion limit
#define MAX_DEPTH 10

// Area light shape types
#define RECTANGLE 1
#define SPHERE 2

// Material types
#define DIFFUSE_REFL 1
#define SPEC_REFL 2
#define SPEC_TRANS 3
#define SPEC_GLASS 4
#define MICROFACET_REFL 5
#define PLASTIC 6
#define DIFFUSE_TRANS 7

// Data structures
struct Ray
{
    vec3 origin;
    vec3 direction;
};

struct Material
{
    vec3 albedo;
    float roughness;
    float eta;  // For transmissive materials
    int type;   // Refer to the #defines above

    // Indices into an array of sampler2Ds that
    // refer to a texture map and/or roughness map.
    // -1 if they aren't used.
    int albedoTex;
    int normalTex;
    int roughnessTex;
};

struct Intersection
{
    float t;
    vec3 nor;
    vec2 uv;
    vec3 Le;  // Emitted light
    int obj_ID;
    Material material;
};

struct Transform
{
    mat4 T;
    mat4 invT;
    mat3 invTransT;
    vec3 scale;
};

struct AreaLight
{
    vec3 Le;
    int ID;

    // RECTANGLE, BOX, SPHERE, or DISC
    // They are all assumed to be "unit size"
    // and are altered from that size by their Transform
    int shapeType;
    Transform transform;
};

struct PointLight
{
    vec3 Le;
    int ID;
    vec3 pos;
};

struct SpotLight
{
    vec3 Le;
    int ID;
    float innerAngle, outerAngle;
    Transform transform;
};

struct Sphere
{
    vec3 pos;
    float radius;

    Transform transform;
    int ID;
    Material material;
};

struct Rectangle
{
    vec3 pos;
    vec3 nor;
    vec2 halfSideLengths;  // Dist from center to horizontal/vertical edge

    Transform transform;
    int ID;
    Material material;
};

struct Box
{
    vec3 minCorner;
    vec3 maxCorner;

    Transform transform;
    int ID;
    Material material;
};

struct Mesh
{
    int triangle_sampler_index;
    int triangle_storage_side_len;
    int num_tris;

    Transform transform;
    int ID;
    Material material;
};

struct Triangle
{
    vec3 pos[3];
    vec3 nor[3];
    vec2 uv[3];
};

// Functions
float AbsDot(vec3 a, vec3 b)
{
    return abs(dot(a, b));
}

float CosTheta(vec3 w)
{
    return w.z;
}

float Cos2Theta(vec3 w)
{
    return w.z * w.z;
}

float AbsCosTheta(vec3 w)
{
    return abs(w.z);
}

float Sin2Theta(vec3 w)
{
    return max(0.f, 1.f - Cos2Theta(w));
}

float SinTheta(vec3 w)
{
    return sqrt(Sin2Theta(w));
}

float TanTheta(vec3 w)
{
    return SinTheta(w) / CosTheta(w);
}

float Tan2Theta(vec3 w)
{
    return Sin2Theta(w) / Cos2Theta(w);
}

float CosPhi(vec3 w)
{
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : clamp(w.x / sinTheta, -1.f, 1.f);
}

float SinPhi(vec3 w)
{
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : clamp(w.y / sinTheta, -1.f, 1.f);
}

float Cos2Phi(vec3 w)
{
    return CosPhi(w) * CosPhi(w);
}

float Sin2Phi(vec3 w)
{
    return SinPhi(w) * SinPhi(w);
}

Ray SpawnRay(vec3 pos, vec3 wi)
{
    return Ray(pos + wi * 0.0001, wi);
}

mat4 translate(vec3 t)
{
    return mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, t.x, t.y, t.z, 1);
}

float radians(float deg)
{
    return deg * PI / 180.f;
}

mat4 rotateX(float rad)
{
    return mat4(1, 0, 0, 0, 0, cos(rad), sin(rad), 0, 0, -sin(rad), cos(rad), 0, 0, 0, 0, 1);
}

mat4 rotateY(float rad)
{
    return mat4(cos(rad), 0, -sin(rad), 0, 0, 1, 0, 0, sin(rad), 0, cos(rad), 0, 0, 0, 0, 1);
}

mat4 rotateZ(float rad)
{
    return mat4(cos(rad), sin(rad), 0, 0, -sin(rad), cos(rad), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
}

mat4 scale(vec3 s)
{
    return mat4(s.x, 0, 0, 0, 0, s.y, 0, 0, 0, 0, s.z, 0, 0, 0, 0, 1);
}

Transform makeTransform(vec3 t, vec3 euler, vec3 s)
{
    mat4 T = translate(t) * rotateX(radians(euler.x)) * rotateY(radians(euler.y))
             * rotateZ(radians(euler.z)) * scale(s);

    return Transform(T, inverse(T), inverse(transpose(mat3(T))), s);
}

bool Refract(vec3 wi, vec3 n, float eta, out vec3 wt)
{
    // Compute cos theta using Snell's law
    float cosThetaI = dot(n, wi);
    float sin2ThetaI = max(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) {
        return false;
    }
    float cosThetaT = sqrt(1 - sin2ThetaT);
    wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

vec3 Faceforward(vec3 n, vec3 v)
{
    return (dot(n, v) < 0.f) ? -n : n;
}

bool SameHemisphere(vec3 w, vec3 wp)
{
    return w.z * wp.z > 0;
}

void coordinateSystem(in vec3 v1, out vec3 v2, out vec3 v3)
{
    if (abs(v1.x) > abs(v1.y)) {
        v2 = vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    } else {
        v2 = vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    }
    v3 = cross(v1, v2);
}

mat3 LocalToWorld(vec3 nor)
{
    vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return mat3(tan, bit, nor);
}

mat3 WorldToLocal(vec3 nor)
{
    return transpose(LocalToWorld(nor));
}

float DistanceSquared(vec3 p1, vec3 p2)
{
    return dot(p1 - p2, p1 - p2);
}

// from ShaderToy https://www.shadertoy.com/view/4tXyWN
uvec2 seed;

float rng()
{
    seed += uvec2(1);
    uvec2 q = 1103515245U * ((seed >> 1U) ^ (seed.yx));
    uint n = 1103515245U * ((q.x) ^ (q.y >> 3U));
    return float(n) * (1.0 / float(0xffffffffU));
}

#define N_TEXTURES 0
#define N_BOXES 0
#define N_RECTANGLES 4
#define N_SPHERES 0
#define N_MESHES 0
#define N_TRIANGLES 0
#define N_LIGHTS 3
#define N_AREA_LIGHTS 3
#define N_POINT_LIGHTS 0
#define N_SPOT_LIGHTS 0

const Rectangle rectangles[N_RECTANGLES] = Rectangle[](Rectangle(vec3(0, 0, 0), vec3(0, 0, 1), vec2(0.5, 0.5), Transform(mat4(80, 0, 0, 0, 0, 1.6633, -7.82518, 0, 0, 0.978147, 0.207913, 0, 0, -31.2, 28.4, 1), mat4(0.0125, 0, 0, 0, 0, 0.0259891, 0.978147, 0, 0, -0.122268, 0.207913, 0, 0, 4.28328, 24.6135, 1), mat3(0.0125, 0, 0, 0, 0.0259891, -0.122268, 0, 0.978147, 0.207913), vec3(80, 8, 1)), 0, Material(vec3(1, 0.395, 0.375), 0.25, -1, 5, -1, -1, -1)),
Rectangle(vec3(0, 0, 0), vec3(0, 0, 1), vec2(0.5, 0.5), Transform(mat4(80, 0, 0, 0, 0, 2.33898, -7.65044, 0, 0, 0.956304, 0.292373, 0, 0, -28.672, 17.848, 1), mat4(0.0125, 0, 0, 0, 0, 0.0365466, 0.956305, 0, 0, -0.119538, 0.292373, 0, 0, 3.18138, 22.2009, 1), mat3(0.0125, 0, 0, 0, 0.0365466, -0.119538, 0, 0.956305, 0.292373), vec3(80, 8, 1)), 1, Material(vec3(1, 1, 0.35), 0.1, -1, 5, -1, -1, -1)),
Rectangle(vec3(0, 0, 0), vec3(0, 0, 1), vec2(0.5, 0.5), Transform(mat4(80, 0, 0, 0, 0, 3.63193, -7.12805, 0, 0, 0.891006, 0.453991, 0, 0, -24.496, 7.368, 1), mat4(0.0125, 0, 0, 0, 0, 0.0567489, 0.891006, 0, 0, -0.111376, 0.453991, 0, 0, 2.21074, 18.4811, 1), mat3(0.0125, 0, 0, 0, 0.0567489, -0.111376, 0, 0.891006, 0.453991), vec3(80, 8, 1)), 2, Material(vec3(0.375, 1, 0.425), 0.05, -1, 5, -1, -1, -1)),
Rectangle(vec3(0, 0, 0), vec3(0, 0, 1), vec2(0.5, 0.5), Transform(mat4(80, 0, 0, 0, 0, 4.98012, -6.26086, 0, 0, 0.782608, 0.622515, 0, 0, -19.16, -1.08, 1), mat4(0.0125, 0, 0, 0, 0, 0.0778144, 0.782608, 0, 0, -0.097826, 0.622515, 0, 0, 1.38527, 15.6671, 1), mat3(0.0125, 0, 0, 0, 0.0778144, -0.097826, 0, 0.782608, 0.622515), vec3(80, 8, 1)), 3, Material(vec3(0.39, 0.33, 0.95), 0.01, -1, 5, -1, -1, -1))
);
const AreaLight areaLights[N_AREA_LIGHTS] = AreaLight[](AreaLight(vec3(96, 96, 96), 4, 2, Transform(mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -28, 0, 0, 1), mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 28, 0, 0, 1), mat3(1, 0, 0, 0, 1, 0, 0, 0, 1), vec3(1, 1, 1))),
AreaLight(vec3(48, 48, 48), 5, 2, Transform(mat4(4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1), mat4(0.25, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 1), mat3(0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25), vec3(4, 4, 4))),
AreaLight(vec3(48, 48, 48), 6, 2, Transform(mat4(8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 28, 0, 0, 1), mat4(0.125, 0, 0, 0, 0, 0.125, 0, 0, 0, 0, 0.125, 0, -3.5, 0, 0, 1), mat3(0.125, 0, 0, 0, 0.125, 0, 0, 0, 0.125), vec3(8, 8, 8)))
);

vec2 PolarToCartesian(float r, float theta)
{
    float x = r * cos(theta);
    float y = r * sin(theta);

    return vec2(x, y);
}

vec3 squareToDiskConcentric(vec2 xi)
{
    vec2 offsetSample = 2.f * xi - vec2(1.f);  // offset to (-1.f, 1.f) range

    float r, theta;

    if (offsetSample.x == 0.f && offsetSample.y == 0.f) {
        return vec3(0.f);
    }

    if (abs(offsetSample.x) > abs(offsetSample.y)) {
        r = offsetSample.x;
        theta = (PI / 4.f) * (offsetSample.y / offsetSample.x);
    } else {
        r = offsetSample.y;
        theta = (PI / 2.f) - ((PI / 4.f) * (offsetSample.x / offsetSample.y));
    }

    vec2 xy = PolarToCartesian(r, theta);

    return vec3(xy, 0.f);
}

// sample
vec3 squareToHemisphereCosine(vec2 xi)
{
    vec3 xy0 = squareToDiskConcentric(xi);
    float x = xy0.x;
    float y = xy0.y;

    float z = sqrt(max(0.f, 1.f - pow(x, 2.f) - pow(y, 2.f)));

    return vec3(x, y, z);
}

float squareToHemisphereCosinePDF(vec3 w)
{
    return CosTheta(w) * INV_PI;
}

vec3 squareToSphereUniform(vec2 xi)
{
    float z = 1.f - (2.f * xi.x);  // map [0, 1] to [-1, 1]

    float r = sqrt(max(0.f, 1.f - pow(z, 2.f)));

    float phi = TWO_PI * xi.y;

    vec2 xy = PolarToCartesian(r, phi);

    return vec3(xy, z);
}

float squareToSphereUniformPDF(vec3 w)
{
    return INV_FOUR_PI;
}

vec3 f_diffuse(vec3 albedo)
{
    // dividing by PI means the integral evaluates to the material's base color.
    return albedo * INV_PI;
}

vec3 Sample_f_diffuse(vec3 albedo,
                      vec2 xi,
                      vec3 nor,
                      out vec3 wiW,
                      out float pdf,
                      out int sampledType)
{
    vec3 wi = squareToHemisphereCosine(xi);
    // Set wiW to a world-space ray direction since wo is in tangent space.
    mat3 worldMat = LocalToWorld(nor);
    wiW = worldMat * wi;

    pdf = squareToHemisphereCosinePDF(wi);  // tangent space pdf

    sampledType = DIFFUSE_REFL;

    return (albedo * INV_PI);
}

vec3 Sample_f_specular_refl(vec3 albedo, vec3 nor, vec3 wo, out vec3 wiW, out int sampledType)
{
    // GLSL reflect assumes incident vector's direction toward origin
    if (wo.z > 0) {
        wo = -wo;
    }
    vec3 localNor = vec3(0.f, 0.f, 1.f);
    vec3 wi = reflect(normalize(wo), normalize(localNor));

    // Set wiW to a world-space ray direction since wo is in tangent space.
    mat3 worldMat = LocalToWorld(nor);
    wiW = worldMat * wi;

    sampledType = SPEC_REFL;

    return albedo / AbsCosTheta(wi);
}

vec3 Sample_f_specular_trans(vec3 albedo, vec3 nor, vec3 wo, out vec3 wiW, out int sampledType)
{
    // Hard-coded to index of refraction of glass
    float etaA = 1.f;
    float etaB = 1.55f;  // currently, ray is travelling into glass

    float etaI;
    float etaT;

    vec3 localNor = vec3(0.f, 0.f, 1.f);

    if (SameHemisphere(wo, vec3(0.f, 0.f, 1.f))) {
        // this is okay for Snell's law
        etaI = etaB;
        etaT = etaA;
    } else {
        // goes against Snell's law assumptions
        etaI = etaA;
        etaT = etaB;

        wo.z = -wo.z;
        localNor = -localNor;
    }

    float eta = etaT / etaI;

    vec3 wt;
    vec3 result;

    bool isReflected = !Refract(wo, localNor, eta, wt);

    mat3 worldMat = LocalToWorld(nor);
    wiW = worldMat * wt;

    if (isReflected) {
        sampledType = SPEC_REFL;
        return vec3(0.f);
    }

    sampledType = SPEC_TRANS;
    return albedo / AbsCosTheta(wt);
}

// dielectric materials are those that can act as an electric insulator
vec3 FresnelDielectricEval(float cosThetaI)
{
    // We will hard-code the indices of refraction to be
    // those of glass
    // currently these assignments go against assumptions of Snell's law / Fresnel's equations
    float etaI = 1.;
    float etaT = 1.55;
    cosThetaI = clamp(cosThetaI, -1.f, 1.f);

    float eta;
    if (cosThetaI > 0.f) {
        // this is actually what we want
        float etaTemp = etaI;
        etaI = etaT;  // incident is glass, so 1.55
        etaT = etaTemp;

        // However, this means cosThetaI was actually cosThetaT in the context of Snell's,
        // and we should now calculate for the real cosThetaI. eta is inverted.
        eta = etaI / etaT;
    } else {
        cosThetaI = -cosThetaI;  // flip to be able to plug in to Snell's.
        eta = etaT / etaI;
    }

    float sin2ThetaI = 1 - pow(cosThetaI, 2.f);
    float sin2ThetaT = sin2ThetaI / pow(eta, 2.f);
    if (sin2ThetaT >= 1.f) {
        return vec3(1.f);
    }
    float cosThetaT = sqrt(max(1.f - sin2ThetaT, 0.f));

    float Er_parl = eta * cosThetaI - cosThetaT;  // E describes amplitude of light waves
    float Ei_parl = eta * cosThetaI + cosThetaT;

    float Er_perp = cosThetaI - eta * cosThetaT;
    float Ei_perp = cosThetaI + eta * cosThetaT;

    float r_parl = Er_parl / Ei_parl;  // r describes power of reflectance
    float r_perp = Er_perp / Ei_perp;

    float r_avg = (pow(r_parl, 2.f) + pow(r_perp, 2.f))
                  / 2.f;  // this describes the average power of reflectance

    return vec3(r_avg);
}

vec3 Sample_f_glass(vec3 albedo, vec3 nor, vec2 xi, vec3 wo, out vec3 wiW, out int sampledType)
{
    float random = rng();
    if (random < 0.5) {
        // Have to double contribution b/c we only sample
        // reflection BxDF half the time
        vec3 R = Sample_f_specular_refl(albedo, nor, wo, wiW, sampledType);
        sampledType = SPEC_REFL;
        return 2 * FresnelDielectricEval(dot(nor, normalize(wiW))) * R;
    } else {
        // Have to double contribution b/c we only sample
        // transmit BxDF half the time
        vec3 T = Sample_f_specular_trans(albedo, nor, wo, wiW, sampledType);
        sampledType = SPEC_TRANS;
        return 2 * (vec3(1.) - FresnelDielectricEval(dot(nor, normalize(wiW)))) * T;
    }
}

// Below are a bunch of functions for handling microfacet materials.
// Don't worry about this for now.
vec3 Sample_wh(vec3 wo, vec2 xi, float roughness)
{
    vec3 wh;

    float cosTheta = 0;
    float phi = TWO_PI * xi[1];
    // We'll only handle isotropic microfacet materials
    float tanTheta2 = roughness * roughness * xi[0] / (1.0f - xi[0]);
    cosTheta = 1 / sqrt(1 + tanTheta2);

    float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));

    wh = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    if (!SameHemisphere(wo, wh)) {
        wh = -wh;
    }

    return wh;
}

float TrowbridgeReitzD(vec3 wh, float roughness)
{
    float tan2Theta = Tan2Theta(wh);
    if (isinf(tan2Theta)) {
        return 0.f;
    }

    float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

    float e = (Cos2Phi(wh) / (roughness * roughness) + Sin2Phi(wh) / (roughness * roughness))
              * tan2Theta;
    return 1 / (PI * roughness * roughness * cos4Theta * (1 + e) * (1 + e));
}

float Lambda(vec3 w, float roughness)
{
    float absTanTheta = abs(TanTheta(w));
    if (isinf(absTanTheta)) {
        return 0.;
    }

    // Compute alpha for direction w
    float alpha = sqrt(Cos2Phi(w) * roughness * roughness + Sin2Phi(w) * roughness * roughness);
    float alpha2Tan2Theta = (roughness * absTanTheta) * (roughness * absTanTheta);
    return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
}

float TrowbridgeReitzG(vec3 wo, vec3 wi, float roughness)
{
    return 1 / (1 + Lambda(wo, roughness) + Lambda(wi, roughness));
}

float TrowbridgeReitzPdf(vec3 wo, vec3 wh, float roughness)
{
    return TrowbridgeReitzD(wh, roughness) * AbsCosTheta(wh);
}

vec3 f_microfacet_refl(vec3 albedo, vec3 wo, vec3 wi, float roughness)
{
    float cosThetaO = AbsCosTheta(wo);
    float cosThetaI = AbsCosTheta(wi);
    vec3 wh = wi + wo;
    // Handle degenerate cases for microfacet reflection
    if (cosThetaI == 0 || cosThetaO == 0) {
        return vec3(0.f);
    }
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) {
        return vec3(0.f);
    }
    wh = normalize(wh);
    // TODO: Handle different Fresnel coefficients
    vec3 F = vec3(1.);  //fresnel->Evaluate(glm::dot(wi, wh));
    float D = TrowbridgeReitzD(wh, roughness);
    float G = TrowbridgeReitzG(wo, wi, roughness);
    return albedo * D * G * F / (4 * cosThetaI * cosThetaO);
}

vec3 Sample_f_microfacet_refl(vec3 albedo,
                              vec3 nor,
                              vec2 xi,
                              vec3 wo,
                              float roughness,
                              out vec3 wiW,
                              out float pdf,
                              out int sampledType)
{
    if (wo.z == 0) {
        return vec3(0.);
    }

    vec3 wh = Sample_wh(wo, xi, roughness);
    vec3 wi = reflect(-wo, wh);
    wiW = LocalToWorld(nor) * wi;
    if (!SameHemisphere(wo, wi)) {
        return vec3(0.f);
    }

    // Compute PDF of _wi_ for microfacet reflection
    pdf = TrowbridgeReitzPdf(wo, wh, roughness) / (4 * dot(wo, wh));
    return f_microfacet_refl(albedo, wo, wi, roughness);
}

vec3 computeAlbedo(Intersection isect)
{
    vec3 albedo = isect.material.albedo;
#if N_TEXTURES
    if (isect.material.albedoTex != -1) {
        albedo *= pow(texture(u_TexSamplers[isect.material.albedoTex], isect.uv).rgb, vec3(2.2f));
    }
#endif
    return albedo;
}

vec3 computeNormal(Intersection isect)
{
    vec3 nor = isect.nor;
#if N_TEXTURES
    if (isect.material.normalTex != -1) {
        vec3 localNor = texture(u_TexSamplers[isect.material.normalTex], isect.uv).rgb;
        vec3 tan, bit;
        coordinateSystem(nor, tan, bit);
        nor = mat3(tan, bit, nor) * localNor;
    }
#endif
    return nor;
}

float computeRoughness(Intersection isect)
{
    float roughness = isect.material.roughness;
#if N_TEXTURES
    if (isect.material.roughnessTex != -1) {
        roughness = texture(u_TexSamplers[isect.material.roughnessTex], isect.uv).r;
    }
#endif
    return roughness;
}

// Computes the overall light scattering properties of a point on a Material,
// given the incoming and outgoing light directions.
vec3 f(Intersection isect, vec3 woW, vec3 wiW)
{
    // Convert the incoming and outgoing light rays from
    // world space to local tangent space
    vec3 nor = computeNormal(isect);
    vec3 wo = WorldToLocal(nor) * woW;
    vec3 wi = WorldToLocal(nor) * wiW;

    // If the outgoing ray is parallel to the surface,
    // we know we can return black b/c the Lambert term
    // in the overall Light Transport Equation will be 0.
    if (wo.z == 0) {
        return vec3(0.f);
    }

    // Since GLSL does not support classes or polymorphism,
    // we have to handle each material type with its own function.
    if (isect.material.type == DIFFUSE_REFL) {
        return f_diffuse(computeAlbedo(isect));
    }
    // As we discussed in class, there is a 0% chance that a randomly
    // chosen wi will be the perfect reflection / refraction of wo,
    // so any specular material will have a BSDF of 0 when wi is chosen
    // independently of the material.
    else if (isect.material.type == SPEC_REFL || isect.material.type == SPEC_TRANS
             || isect.material.type == SPEC_GLASS) {
        return vec3(0.);
    } else if (isect.material.type == MICROFACET_REFL) {
        return f_microfacet_refl(computeAlbedo(isect), wo, wi, computeRoughness(isect));
    }
    // Default case, unhandled material
    else {
        return vec3(1, 0, 1);
    }
}

// Sample_f() returns the same values as f(), but importantly it
// only takes in a wo. Note that wiW is declared as an "out vec3";
// this means the function is intended to compute and write a wi
// in world space (the trailing "W" indicates world space).
// In other words, Sample_f() evaluates the BSDF *after* generating
// a wi based on the Intersection's material properties, allowing
// us to bias our wi samples in a way that gives more consistent
// light scattered along wo.
vec3 Sample_f(Intersection isect, vec3 woW, vec2 xi, out vec3 wiW, out float pdf, out int sampledType)
{
    // Convert wo to local space from world space.
    // The various Sample_f()s output a wi in world space,
    // but assume wo is in local space.
    vec3 nor = computeNormal(isect);
    vec3 wo = WorldToLocal(nor) * woW;

    if (isect.material.type == DIFFUSE_REFL) {
        return Sample_f_diffuse(computeAlbedo(isect), xi, nor, wiW, pdf, sampledType);
    } else if (isect.material.type == SPEC_REFL) {
        pdf = 1.;
        return Sample_f_specular_refl(computeAlbedo(isect), nor, wo, wiW, sampledType);
    } else if (isect.material.type == SPEC_TRANS) {
        pdf = 1.;
        return Sample_f_specular_trans(computeAlbedo(isect), nor, wo, wiW, sampledType);
    } else if (isect.material.type == SPEC_GLASS) {
        pdf = 1.;
        return Sample_f_glass(computeAlbedo(isect), nor, xi, wo, wiW, sampledType);
    } else if (isect.material.type == MICROFACET_REFL) {
        return Sample_f_microfacet_refl(computeAlbedo(isect),
                                        nor,
                                        xi,
                                        wo,
                                        computeRoughness(isect),
                                        wiW,
                                        pdf,
                                        sampledType);
    } else if (isect.material.type == PLASTIC) {
        return vec3(1, 0, 1);
    }
    // Default case, unhandled material
    else {
        return vec3(1, 0, 1);
    }
}

// Compute the PDF of wi with respect to wo and the intersection's
// material properties.
float Pdf(Intersection isect, vec3 woW, vec3 wiW)
{
    vec3 nor = computeNormal(isect);
    vec3 wo = WorldToLocal(nor) * woW;
    vec3 wi = WorldToLocal(nor) * wiW;

    if (wo.z == 0) {
        return 0.;  // The cosine of this vector would be zero
    }

    if (isect.material.type == DIFFUSE_REFL) {
        return squareToHemisphereCosinePDF(wi);
    } else if (isect.material.type == SPEC_REFL || isect.material.type == SPEC_TRANS
               || isect.material.type == SPEC_GLASS) {
        return 0.;
    } else if (isect.material.type == MICROFACET_REFL) {
        vec3 wh = normalize(wo + wi);
        return TrowbridgeReitzPdf(wo, wh, computeRoughness(isect)) / (4 * dot(wo, wh));
    }
    // Default case, unhandled material
    else {
        return 0.;
    }
}

// optimized algorithm for solving quadratic equations developed by Dr. Po-Shen Loh -> https://youtu.be/XKBX0r3J-9Y
// Adapted to root finding (ray t0/t1) for all quadric shapes (sphere, ellipsoid, cylinder, cone, etc.) by Erich Loftis
void solveQuadratic(float A, float B, float C, out float t0, out float t1)
{
    float invA = 1.0 / A;
    B *= invA;
    C *= invA;
    float neg_halfB = -B * 0.5;
    float u2 = neg_halfB * neg_halfB - C;
    float u = u2 < 0.0 ? neg_halfB = 0.0 : sqrt(u2);
    t0 = neg_halfB - u;
    t1 = neg_halfB + u;
}

vec2 sphereUVMap(vec3 p)
{
    float phi = atan(p.z, p.x);
    if (phi < 0) {
        phi += TWO_PI;
    }
    float theta = acos(p.y);
    return vec2(1 - phi / TWO_PI, 1 - theta / PI);
}

float sphereIntersect(Ray ray, float radius, vec3 pos, out vec3 localNor, out vec2 out_uv, mat4 invT)
{
    ray.origin = vec3(invT * vec4(ray.origin, 1.));
    ray.direction = vec3(invT * vec4(ray.direction, 0.));
    float t0, t1;
    vec3 diff = ray.origin - pos;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(ray.direction, diff);
    float c = dot(diff, diff) - (radius * radius);
    solveQuadratic(a, b, c, t0, t1);
    localNor = t0 > 0.0 ? ray.origin + t0 * ray.direction : ray.origin + t1 * ray.direction;
    localNor = normalize(localNor);
    out_uv = sphereUVMap(localNor);
    return t0 > 0.0 ? t0 : t1 > 0.0 ? t1 : INFINITY;
}

float planeIntersect(vec4 pla, vec3 rayOrigin, vec3 rayDirection, mat4 invT)
{
    rayOrigin = vec3(invT * vec4(rayOrigin, 1.));
    rayDirection = vec3(invT * vec4(rayDirection, 0.));
    vec3 n = pla.xyz;
    float denom = dot(n, rayDirection);

    vec3 pOrO = (pla.w * n) - rayOrigin;
    float result = dot(pOrO, n) / denom;
    return (result > 0.0) ? result : INFINITY;
}

float rectangleIntersect(vec3 pos,
                         vec3 normal,
                         float radiusU,
                         float radiusV,
                         vec3 rayOrigin,
                         vec3 rayDirection,
                         out vec2 out_uv,
                         mat4 invT)
{
    rayOrigin = vec3(invT * vec4(rayOrigin, 1.));
    rayDirection = vec3(invT * vec4(rayDirection, 0.));
    float dt = dot(-normal, rayDirection);
    // use the following for one-sided rectangle
    if (dt < 0.0) {
        return INFINITY;
    }
    float t = dot(-normal, pos - rayOrigin) / dt;
    if (t < 0.0) {
        return INFINITY;
    }

    vec3 hit = rayOrigin + rayDirection * t;
    vec3 vi = hit - pos;
    vec3 U = normalize(cross(abs(normal.y) < 0.9 ? vec3(0, 1, 0) : vec3(1, 0, 0), normal));
    vec3 V = cross(normal, U);

    out_uv = vec2(dot(U, vi) / length(U), dot(V, vi) / length(V));
    out_uv = out_uv + vec2(0.5, 0.5);

    return (abs(dot(U, vi)) > radiusU || abs(dot(V, vi)) > radiusV) ? INFINITY : t;
}

float boxIntersect(vec3 minCorner,
                   vec3 maxCorner,
                   mat4 invT,
                   mat3 invTransT,
                   vec3 rayOrigin,
                   vec3 rayDirection,
                   out vec3 normal,
                   out bool isRayExiting,
                   out vec2 out_uv)
{
    rayOrigin = vec3(invT * vec4(rayOrigin, 1.));
    rayDirection = vec3(invT * vec4(rayDirection, 0.));
    vec3 invDir = 1.0 / rayDirection;
    vec3 near = (minCorner - rayOrigin) * invDir;
    vec3 far = (maxCorner - rayOrigin) * invDir;
    vec3 tmin = min(near, far);
    vec3 tmax = max(near, far);
    float t0 = max(max(tmin.x, tmin.y), tmin.z);
    float t1 = min(min(tmax.x, tmax.y), tmax.z);
    if (t0 > t1) {
        return INFINITY;
    }
    if (t0 > 0.0)  // if we are outside the box
    {
        normal = -sign(rayDirection) * step(tmin.yzx, tmin) * step(tmin.zxy, tmin);
        normal = normalize(invTransT * normal);
        isRayExiting = false;
        vec3 p = t0 * rayDirection + rayOrigin;
        p = (p - minCorner) / (maxCorner - minCorner);
        out_uv = p.xy;
        return t0;
    }
    if (t1 > 0.0)  // if we are inside the box
    {
        normal = -sign(rayDirection) * step(tmax, tmax.yzx) * step(tmax, tmax.zxy);
        normal = normalize(invTransT * normal);
        isRayExiting = true;
        vec3 p = t1 * rayDirection + rayOrigin;
        p = (p - minCorner) / (maxCorner - minCorner);
        out_uv = p.xy;
        return t1;
    }
    return INFINITY;
}

// Möller–Trumbore intersection
float triangleIntersect(vec3 p0, vec3 p1, vec3 p2, vec3 rayOrigin, vec3 rayDirection)
{
    const float EPSILON = 0.0000001;
    vec3 edge1, edge2, h, s, q;
    float a, f, u, v;
    edge1 = p1 - p0;
    edge2 = p2 - p0;
    h = cross(rayDirection, edge2);
    a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) {
        return INFINITY;  // This ray is parallel to this triangle.
    }
    f = 1.0 / a;
    s = rayOrigin - p0;
    u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return INFINITY;
    }
    q = cross(s, edge1);
    v = f * dot(rayDirection, q);
    if (v < 0.0 || u + v > 1.0) {
        return INFINITY;
    }
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * dot(edge2, q);
    if (t > EPSILON) {
        return t;
    } else {  // This means that there is a line intersection but not a ray intersection.
        return INFINITY;
    }
}

vec3 barycentric(vec3 p, vec3 t1, vec3 t2, vec3 t3)
{
    vec3 edge1 = t2 - t1;
    vec3 edge2 = t3 - t2;
    float S = length(cross(edge1, edge2));

    edge1 = p - t2;
    edge2 = p - t3;
    float S1 = length(cross(edge1, edge2));

    edge1 = p - t1;
    edge2 = p - t3;
    float S2 = length(cross(edge1, edge2));

    edge1 = p - t1;
    edge2 = p - t2;
    float S3 = length(cross(edge1, edge2));

    return vec3(S1 / S, S2 / S, S3 / S);
}

#if N_MESHES
float meshIntersect(int mesh_id,
                    vec3 rayOrigin,
                    vec3 rayDirection,
                    out vec3 out_nor,
                    out vec2 out_uv,
                    mat4 invT)
{
    rayOrigin = vec3(invT * vec4(rayOrigin, 1.));
    rayDirection = vec3(invT * vec4(rayDirection, 0.));

    int sampIdx = 0;  // meshes[mesh_id].triangle_sampler_index;

    float t = INFINITY;

    // Iterate over each triangle, and
    // convert it to a pixel coordinate
    for (int i = 0; i < meshes[mesh_id].num_tris; ++i) {
        // pos0, pos1, pos2, nor0, nor1, nor2, uv0, uv1, uv2
        // Each triangle takes up 9 pixels
        Triangle tri;
        int first_pixel = i * 9;
        // Positions
        for (int p = first_pixel; p < first_pixel + 3; ++p) {
            int row = int(floor(float(p) / meshes[mesh_id].triangle_storage_side_len));
            int col = p - row * meshes[mesh_id].triangle_storage_side_len;

            tri.pos[p - first_pixel]
                = texelFetch(u_TriangleStorageSamplers[sampIdx], ivec2(col, row), 0).rgb;
        }
        first_pixel += 3;
        // Normals
        for (int n = first_pixel; n < first_pixel + 3; ++n) {
            int row = int(floor(float(n) / meshes[mesh_id].triangle_storage_side_len));
            int col = n - row * meshes[mesh_id].triangle_storage_side_len;

            tri.nor[n - first_pixel]
                = texelFetch(u_TriangleStorageSamplers[sampIdx], ivec2(col, row), 0).rgb;
        }
        first_pixel += 3;
        // UVs
        for (int v = first_pixel; v < first_pixel + 3; ++v) {
            int row = int(floor(float(v) / meshes[mesh_id].triangle_storage_side_len));
            int col = v - row * meshes[mesh_id].triangle_storage_side_len;

            tri.uv[v - first_pixel]
                = texelFetch(u_TriangleStorageSamplers[sampIdx], ivec2(col, row), 0).rg;
        }

        float d = triangleIntersect(tri.pos[0], tri.pos[1], tri.pos[2], rayOrigin, rayDirection);
        if (d < t) {
            t = d;
            vec3 p = rayOrigin + t * rayDirection;
            vec3 baryWeights = barycentric(p, tri.pos[0], tri.pos[1], tri.pos[2]);
            out_nor = baryWeights[0] * tri.nor[0] + baryWeights[1] * tri.nor[1]
                      + baryWeights[2] * tri.nor[2];
            out_uv = baryWeights[0] * tri.uv[0] + baryWeights[1] * tri.uv[1]
                     + baryWeights[2] * tri.uv[2];
        }
    }

    return t;
}
#endif

Intersection sceneIntersect(Ray ray)
{
    float t = INFINITY;
    Intersection result;
    result.t = INFINITY;

#if N_RECTANGLES
    for (int i = 0; i < N_RECTANGLES; ++i) {
        vec2 uv;
        Rectangle rect = rectangles[i];
        float d = rectangleIntersect(rect.pos,
                                     rect.nor,
                                     rect.halfSideLengths.x,
                                     rect.halfSideLengths.y,
                                     ray.origin,
                                     ray.direction,
                                     uv,
                                     rect.transform.invT);
        if (d < t) {
            t = d;
            result.t = t;
            result.nor = normalize(rect.transform.invTransT * rect.nor);
            result.uv = uv;
            result.Le = vec3(0, 0, 0);
            result.obj_ID = rect.ID;
            result.material = rect.material;
        }
    }
#endif
#if N_BOXES
    for (int i = 0; i < N_BOXES; ++i) {
        vec3 nor;
        bool isExiting;
        vec2 uv;
        Box b = boxes[i];
        float d = boxIntersect(b.minCorner,
                               b.maxCorner,
                               b.transform.invT,
                               b.transform.invTransT,
                               ray.origin,
                               ray.direction,
                               nor,
                               isExiting,
                               uv);
        if (d < t) {
            t = d;
            result.t = t;
            result.nor = nor;
            result.Le = vec3(0, 0, 0);
            result.obj_ID = b.ID;
            result.material = b.material;
            result.uv = uv;
        }
    }
#endif
#if N_SPHERES
    for (int i = 0; i < N_SPHERES; ++i) {
        vec3 nor;
        bool isExiting;
        vec3 localNor;
        vec2 uv;
        Sphere s = spheres[i];
        float d = sphereIntersect(ray, s.radius, s.pos, localNor, uv, s.transform.invT);
        if (d < t) {
            t = d;
            vec3 p = ray.origin + t * ray.direction;
            result.t = t;
            result.nor = normalize(s.transform.invTransT * localNor);
            result.Le = vec3(0, 0, 0);
            result.uv = uv;
            result.obj_ID = s.ID;
            result.material = s.material;
        }
    }
#endif
#if N_MESHES
    for (int i = 0; i < N_MESHES; ++i) {
        vec3 nor;
        vec2 uv;
        float d = meshIntersect(i, ray.origin, ray.direction, nor, uv, meshes[i].transform.invT);

        if (d < t) {
            t = d;
            result.t = t;
            result.nor = nor;
            result.uv = uv;
            result.Le = vec3(0, 0, 0);
            result.obj_ID = meshes[i].ID;
            result.material = meshes[i].material;
        }
    }
#endif
#if N_AREA_LIGHTS
    for (int i = 0; i < N_AREA_LIGHTS; ++i) {
        AreaLight l = areaLights[i];
        int shapeType = l.shapeType;
        if (shapeType == RECTANGLE) {
            vec3 pos = vec3(0, 0, 0);
            vec3 nor = vec3(0, 0, 1);
            vec2 halfSideLengths = vec2(0.5, 0.5);
            vec2 uv;
            float d = rectangleIntersect(pos,
                                         nor,
                                         halfSideLengths.x,
                                         halfSideLengths.y,
                                         ray.origin,
                                         ray.direction,
                                         uv,
                                         l.transform.invT);
            if (d < t) {
                t = d;
                result.t = t;
                result.nor = normalize(l.transform.invTransT * vec3(0, 0, 1));
                result.Le = l.Le;
                result.obj_ID = l.ID;
            }
        } else if (shapeType == SPHERE) {
            vec3 pos = vec3(0, 0, 0);
            float radius = 1.;
            mat4 invT = l.transform.invT;
            vec3 localNor;
            vec2 uv;
            float d = sphereIntersect(ray, radius, pos, localNor, uv, invT);
            if (d < t) {
                t = d;
                result.t = t;
                result.nor = normalize(l.transform.invTransT * localNor);
                result.Le = l.Le;
                result.obj_ID = l.ID;
            }
        }
    }
#endif
#if N_TEXTURES
    if (result.material.normalTex != -1) {
        vec3 localNor = texture(u_TexSamplers[result.material.normalTex], result.uv).rgb;
        localNor = localNor * 2. - vec3(1.);
        vec3 tan, bit;
        coordinateSystem(result.nor, tan, bit);
        result.nor = mat3(tan, bit, result.nor) * localNor;
    }
#endif
    return result;
}

Intersection areaLightIntersect(AreaLight light, Ray ray)
{
    Intersection result;
    result.t = INFINITY;
#if N_AREA_LIGHTS
    int shapeType = light.shapeType;
    if (shapeType == RECTANGLE) {
        vec3 pos = vec3(0, 0, 0);
        vec3 nor = vec3(0, 0, 1);
        vec2 halfSideLengths = vec2(0.5, 0.5);
        vec2 uv;
        float d = rectangleIntersect(pos,
                                     nor,
                                     halfSideLengths.x,
                                     halfSideLengths.y,
                                     ray.origin,
                                     ray.direction,
                                     uv,
                                     light.transform.invT);
        result.t = d;
        result.nor = normalize(light.transform.invTransT * vec3(0, 0, 1));
        result.Le = light.Le;
        result.obj_ID = light.ID;
    } else if (shapeType == SPHERE) {
        vec3 pos = vec3(0, 0, 0);
        float radius = 1.;
        mat4 invT = light.transform.invT;
        vec3 localNor;
        vec2 uv;
        float d = sphereIntersect(ray, radius, pos, localNor, uv, invT);
        result.t = d;
        result.nor = normalize(light.transform.invTransT * localNor);
        result.Le = light.Le;
        result.obj_ID = light.ID;
    }
#endif
    return result;
}

vec2 normalize_uv = vec2(0.1591, 0.3183);

vec2 sampleSphericalMap(vec3 v)
{
    // U is in the range [-PI, PI], V is [-PI/2, PI/2]
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    // Convert UV to [-0.5, 0.5] in U&V
    uv *= normalize_uv;
    // Convert UV to [0, 1]
    uv += 0.5;
    return uv;
}

vec3 sampleFromInsideSphere(vec2 xi, out float pdf)
{
    //    Point3f pObj = WarpFunctions::squareToSphereUniform(xi);

    //    Intersection it;
    //    it.normalGeometric = glm::normalize( transform.invTransT() *pObj );
    //    it.point = Point3f(transform.T() * glm::vec4(pObj.x, pObj.y, pObj.z, 1.0f));

    //    *pdf = 1.0f / Area();

    //    return it;
    return vec3(0.);
}

#if N_AREA_LIGHTS
vec3 DirectSampleAreaLight(int idx,
                           vec3 view_point,
                           vec3 view_nor,
                           int num_lights,
                           out vec3 wiW,
                           out float pdf)
{
    AreaLight light = areaLights[idx];
    int type = light.shapeType;
    Ray shadowRay;

    if (type == RECTANGLE) {
        Transform lightXform = light.transform;

        vec4 point4d = vec4(rng() * 2.f - 1.f, rng() * 2.f - 1.f, 0.f, 1.f);
        vec3 p = (lightXform.T * point4d).xyz;

        vec3 lightNor = normalize(lightXform.invTransT * vec3(0.f, 0.f, 1.f));
        float cosTheta = dot(lightNor, -normalize(p - view_point));

        if (cosTheta <= 0.f) {
            pdf = 0.f;
            return vec3(0.f);
        }

        pdf = 1.f / (2.f * lightXform.scale.x * 2.f * lightXform.scale.y);  // 1 / SurfaceArea
        float r = distance(p, view_point);

        pdf = pdf * r * r / cosTheta;

        wiW = normalize(p - view_point);

        shadowRay = SpawnRay(view_point, wiW);
        Intersection shadowIsect = sceneIntersect(shadowRay);
        if (shadowIsect.obj_ID != light.ID) {
            return vec3(0.f);
        }

        return light.Le * float(num_lights);
    } else if (type == SPHERE) {
        Transform tr = light.transform;

        vec2 xi = vec2(rng(), rng());

        vec3 center = vec3(tr.T * vec4(0., 0., 0., 1.));
        vec3 centerToRef = normalize(center - view_point);
        vec3 tan, bit;

        coordinateSystem(centerToRef, tan, bit);

        vec3 pOrigin;
        if (dot(center - view_point, view_nor) > 0) {
            pOrigin = view_point + view_nor * RayEpsilon;
        } else {
            pOrigin = view_point - view_nor * RayEpsilon;
        }

        // Inside the sphere
        if (dot(pOrigin - center, pOrigin - center) <= 1.f) {  // Radius is 1, so r^2 is also 1
            return sampleFromInsideSphere(xi, pdf);
        }

        float sinThetaMax2 = 1
                             / dot(view_point - center, view_point - center);  // Again, radius is 1
        float cosThetaMax = sqrt(max(0.0f, 1.0f - sinThetaMax2));
        float cosTheta = (1.0f - xi.x) + xi.x * cosThetaMax;
        float sinTheta = sqrt(max(0.f, 1.0f - cosTheta * cosTheta));
        float phi = xi.y * TWO_PI;

        float dc = distance(view_point, center);
        float ds = dc * cosTheta - sqrt(max(0.0f, 1 - dc * dc * sinTheta * sinTheta));

        float cosAlpha = (dc * dc + 1 - ds * ds) / (2 * dc * 1);
        float sinAlpha = sqrt(max(0.0f, 1.0f - cosAlpha * cosAlpha));

        vec3 nObj = sinAlpha * cos(phi) * -tan + sinAlpha * sin(phi) * -bit
                    + cosAlpha * -centerToRef;
        vec3 pObj = vec3(nObj);  // Would multiply by radius, but it is always 1 in object space

        shadowRay = SpawnRay(view_point, normalize(vec3(tr.T * vec4(pObj, 1.0f)) - view_point));
        wiW = shadowRay.direction;
        pdf = 1.0f / (TWO_PI * (1 - cosThetaMax));
        pdf /= tr.scale.x * tr.scale.x;
        return float(num_lights) * light.Le;
    }

    Intersection isect = sceneIntersect(shadowRay);
    if (isect.obj_ID == light.ID) {
        // Multiply by N+1 to account for sampling it 1/(N+1) times.
        // +1 because there's also the environment light
        return float(num_lights) * light.Le;
    }
}
#endif

#if N_POINT_LIGHTS
vec3 DirectSamplePointLight(int idx, vec3 view_point, int num_lights, out vec3 wiW, out float pdf)
{
    PointLight light = pointLights[idx];

    wiW = normalize(light.pos - view_point);

    pdf = 1.f;

    Ray shadowRay = SpawnRay(view_point, normalize(light.pos - view_point));
    Intersection shadowIsect = sceneIntersect(shadowRay);

    float dist = distance(view_point, light.pos);
    if (shadowIsect.t <= dist) {
        return vec3(0.f);
    }

    return light.Le / (dist * dist) * float(num_lights);
}
#endif

#if N_SPOT_LIGHTS
vec3 DirectSampleSpotLight(int idx, vec3 view_point, int num_lights, out vec3 wiW, out float pdf)
{
    SpotLight light = spotLights[idx];
    vec3 lightPos = (light.transform.T * vec4(0.f, 0.f, 0.f, 1.f)).xyz;

    wiW = normalize(lightPos - view_point);

    vec3 wi = vec3(light.transform.invT * vec4(wiW, 0.f)).xyz;

    pdf = 1.f;

    float cosTheta = abs(wi.z);
    float cosOuter = cos(radians(light.outerAngle));
    float cosInner = cos(radians(light.innerAngle));

    if (cosTheta < cosOuter) {
        pdf = 0.f;
        return vec3(0.f);
    }

    Ray shadowRay = SpawnRay(view_point, wiW);
    Intersection shadowIsect = sceneIntersect(shadowRay);

    float dist = distance(view_point, lightPos);
    if (shadowIsect.t <= dist) {
        return vec3(0.f);
    }

    float falloff = 1.f;
    if (cosTheta < cosInner) {
        falloff = smoothstep(cosOuter, cosInner, cosTheta);
    }

    return (falloff * light.Le) / (dist * dist) * float(num_lights);
}
#endif

vec3 Sample_Li(vec3 view_point, vec3 nor, out vec3 wiW, out float pdf)
{
    // Choose a random light from among all of the
    // light sources in the scene, including the environment light
    int num_lights = N_LIGHTS;

#define ENV_MAP 0
#if ENV_MAP
    int num_lights = N_LIGHTS + 1;
#endif
    int randomLightIdx = int(rng() * num_lights);

    // Chose an area light
    if (randomLightIdx < N_AREA_LIGHTS) {
#if N_AREA_LIGHTS
        return DirectSampleAreaLight(randomLightIdx, view_point, nor, num_lights, wiW, pdf);
#endif
    }
    // Chose a point light
    else if (randomLightIdx < N_AREA_LIGHTS + N_POINT_LIGHTS) {
#if N_POINT_LIGHTS
        return DirectSamplePointLight(randomLightIdx - N_AREA_LIGHTS,
                                      view_point,
                                      num_lights,
                                      wiW,
                                      pdf);
#endif
    }
    // Chose a spot light
    else if (randomLightIdx < N_AREA_LIGHTS + N_POINT_LIGHTS + N_SPOT_LIGHTS) {
#if N_SPOT_LIGHTS
        return DirectSampleSpotLight(randomLightIdx - N_AREA_LIGHTS - N_POINT_LIGHTS,
                                     view_point,
                                     num_lights,
                                     wiW,
                                     pdf);
#endif
    }
    // Chose the environment light
    else {
        // TODO
    }
    return vec3(0.);
}

vec3 Sample_Li(vec3 view_point,
               vec3 nor,
               out vec3 wiW,
               out float pdf,
               out int chosenLightIdx,
               out int chosenLightID)
{
    // Choose a random light from among all of the
    // light sources in the scene, including the environment light
    int num_lights = N_LIGHTS;
#define ENV_MAP 0
#if ENV_MAP
    int num_lights = N_LIGHTS + 1;
#endif
    int randomLightIdx = int(rng() * num_lights);
    chosenLightIdx = randomLightIdx;
    // Chose an area light
    if (randomLightIdx < N_AREA_LIGHTS) {
#if N_AREA_LIGHTS
        AreaLight a = areaLights[chosenLightIdx];
        chosenLightID = a.ID;
        return DirectSampleAreaLight(randomLightIdx, view_point, nor, num_lights, wiW, pdf);
#endif
    }
    // Chose a point light
    else if (randomLightIdx < N_AREA_LIGHTS + N_POINT_LIGHTS) {
#if N_POINT_LIGHTS
        PointLight p = pointLights[randomLightIdx - N_AREA_LIGHTS];
        chosenLightID = p.ID;
        return DirectSamplePointLight(randomLightIdx - N_AREA_LIGHTS,
                                      view_point,
                                      num_lights,
                                      wiW,
                                      pdf);
#endif
    }
    // Chose a spot light
    else if (randomLightIdx < N_AREA_LIGHTS + N_POINT_LIGHTS + N_SPOT_LIGHTS) {
#if N_SPOT_LIGHTS
        SpotLight s = spotLights[randomLightIdx - N_AREA_LIGHTS - N_POINT_LIGHTS];
        chosenLightID = s.ID;
        return DirectSampleSpotLight(randomLightIdx - N_AREA_LIGHTS - N_POINT_LIGHTS,
                                     view_point,
                                     num_lights,
                                     wiW,
                                     pdf);
#endif
    }
    // Chose the environment light
    else {
        chosenLightID = -1;
        // TODO
    }
    return vec3(0.);
}

float UniformConePdf(float cosThetaMax)
{
    return 1 / (2 * PI * (1 - cosThetaMax));
}

float SpherePdf(Intersection ref, vec3 p, vec3 wi, Transform transform, float radius)
{
    vec3 nor = ref.nor;
    vec3 pCenter = (transform.T * vec4(0, 0, 0, 1)).xyz;
    // Return uniform PDF if point is inside sphere
    vec3 pOrigin = p + nor * 0.0001;
    // If inside the sphere
    if (DistanceSquared(pOrigin, pCenter) <= radius * radius) {
        //        return Shape::Pdf(ref, wi);
        // To be provided later
        return 0.f;
    }

    // Compute general sphere PDF
    float sinThetaMax2 = radius * radius / DistanceSquared(p, pCenter);
    float cosThetaMax = sqrt(max(0.f, 1.f - sinThetaMax2));
    return UniformConePdf(cosThetaMax);
}

float Pdf_Li(vec3 view_point, vec3 nor, vec3 wiW, int chosenLightIdx)
{
    Ray ray = SpawnRay(view_point, wiW);

    // Area light
    if (chosenLightIdx < N_AREA_LIGHTS) {
#if N_AREA_LIGHTS
        AreaLight light = areaLights[chosenLightIdx];
        Intersection isect = areaLightIntersect(light, ray);
        if (isect.t == INFINITY) {
            // If doesn't intersect anything, 0 PDF
            return 0.;
        }

        if (isect.obj_ID != light.ID) {
            return 0.;  // didn't intersect this light
        }

        vec3 light_point = ray.origin + isect.t * wiW;
        int type = light.shapeType;
        if (type == RECTANGLE) {
            float surfaceArea = 4.f * light.transform.scale.x * light.transform.scale.y;
            float r = distance(light_point, view_point);
            float cosTheta = dot(nor, wiW);

            return (r * r) / (cosTheta * surfaceArea);
        } else if (type == SPHERE) {
            return SpherePdf(isect, light_point, wiW, light.transform, 1.f);
        }
#endif
    }
    // Point light or spot light
    else if (chosenLightIdx < N_AREA_LIGHTS + N_POINT_LIGHTS
             || chosenLightIdx < N_AREA_LIGHTS + N_POINT_LIGHTS + N_SPOT_LIGHTS) {
        return 0;  // No chance of hitting a point in space
    }
    // Env map
    else {
        // TODO
        return 0.f;
    }
}

float PowerHeuristic(int nf, float fPdf, int ng, float gPdf)
{
    float fFit = nf * fPdf;
    float gFit = ng * gPdf;

    return (fFit * fFit) / ((fFit * fFit) + (gFit * gFit));
}

vec3 Direct_MIS(Intersection isect, vec3 woW)
{
    return vec3(0.f);
}

const float FOVY = 19.5f * PI / 180.0;

Ray rayCast()
{
    vec2 offset = vec2(rng(), rng());
    vec2 ndc = (vec2(gl_FragCoord.xy) + offset) / vec2(u_ScreenDims);
    ndc = ndc * 2.f - vec2(1.f);

    float aspect = u_ScreenDims.x / u_ScreenDims.y;
    vec3 ref = u_Eye + u_Forward;
    vec3 V = u_Up * tan(FOVY * 0.5);
    vec3 H = u_Right * tan(FOVY * 0.5) * aspect;
    vec3 p = ref + H * ndc.x + V * ndc.y;

    return Ray(u_Eye, normalize(p - u_Eye));
}

// Procedure:
// 1. Check where ray intersects. Account for if hit light or nothing.
// 2. Compute LTE, where `ray` is the incoming ray.
// 3. Sample a new ray bounce and iterate again.

// Find one Li using an iterative form of raytracing.
vec3 Li_Naive(Ray ray)
{
    vec3 Lo = vec3(0.f);
    // keeps track of the light energy being passed at each bounce of the ray.
    vec3 throughput = vec3(1.f);  // necessary for when surfaces can be emissive as well.

    for (int i = 0; i < MAX_DEPTH; i++) {
        Intersection isect = sceneIntersect(ray);

        if (isect.t == INFINITY) {
            break;
        }

        if (length(isect.Le) > 0.f) {
            Lo += isect.Le * throughput;
            break;
        }

        vec3 woW = -ray.direction;     // in
        vec2 xi = vec2(rng(), rng());  // in

        vec3 wiW;                      // out
        float pdf;                     // out
        int sampledType;               // out

        vec3 bsdf = Sample_f(isect, woW, xi, wiW, pdf, sampledType);

        if (pdf <= 0.f) {
            break;  // don't want any NaN issues
        }

        float lambertTerm = max(0.f, AbsDot(wiW, isect.nor));
        vec3 thisIterThroughput = (bsdf * lambertTerm) / pdf;

        throughput *= thisIterThroughput;

        // generate next ray
        vec3 pPrime = ray.origin + (ray.direction * isect.t);
        ray = SpawnRay(pPrime + (isect.nor * RayEpsilon), wiW);
    }

    return Lo;
}

vec3 Li_Direct_Simple(Ray ray)
{
    vec3 Lo = vec3(0.f);

    Intersection isect = sceneIntersect(ray);

    if (isect.t == INFINITY) {
        return Lo;
    }

    if (length(isect.Le) > 0.f) {
        return isect.Le;
    }

    // world-space position of point that is intersected
    vec3 view_point = ray.origin + (isect.t * ray.direction);  // in
    vec3 nor = isect.nor;

    vec3 wiW;   // out
    float pdf;  // out

    vec3 Li = Sample_Li(view_point, nor, wiW, pdf);

    float lambertTerm = max(0.f, AbsDot(wiW, nor));

    vec3 woW = -ray.direction;

    vec3 bsdf = f(isect, woW, wiW);  // get bsdf from normal f() function

    Lo = (bsdf * Li * lambertTerm) / pdf;

    if (any(isnan(Lo)) || pdf <= 0.f) {
        return vec3(0.f);  // Return black if any values will cause problem
    }

    return Lo;
}

vec3 Li_DirectMIS(Ray ray)
{
    // variables to find
    vec3 Lo = vec3(0.f);

    vec3 Lo_Light = vec3(0.f);
    vec3 Lo_Bsdf = vec3(0.f);

    Intersection isect = sceneIntersect(ray);

    if (isect.t == INFINITY) {
        return vec3(0.f);  // didn't hit anything
    } else if (length(isect.Le) > 0.f) {
        return isect.Le;   // hit a light directly
    }

    // general variables
    vec3 view_point = ray.origin + (isect.t * ray.direction);
    vec3 nor = isect.nor;
    vec3 woW = -ray.direction;

    // variables for direct light ray
    vec3 wiW_Light;      // out
    float pdf_LL;        // out; PDF of light-sampled ray wrt light
    int chosenLightIdx;  // out
    int chosenLightID;   // out

    vec3 Li_Light = Sample_Li(view_point, nor, wiW_Light, pdf_LL, chosenLightIdx, chosenLightID);

    vec3 f_Light = f(isect, woW, wiW_Light);

    float cosTheta_Light = max(0.f, AbsDot(wiW_Light, nor));
    float pdf_LF = Pdf(isect, woW, wiW_Light);  // PDF of light-sampled ray wrt BSDF

    if (pdf_LL <= 0.f) {
        Lo_Light = vec3(0.f);
    } else {
        Lo_Light = Li_Light * f_Light * cosTheta_Light / pdf_LL;
    }

    // variables for BSDF ray
    vec2 xi = vec2(rng(), rng());  // in
    vec3 wiW_Bsdf;                 // out
    float pdf_FF;                  // out; PDF of BSDF-sampled ray wrt BSDF
    float sampledType;             // out

    vec3 f_Bsdf = Sample_f(isect, woW, xi, wiW_Bsdf, pdf_FF, sampledType);

    vec3 Li_Bsdf = vec3(0.f);
    float cosTheta_Bsdf = max(0.f, AbsDot(wiW_Bsdf, nor));
    Ray foundRay = SpawnRay(view_point, wiW_Bsdf);
    Intersection foundIsect = sceneIntersect(foundRay);

    if (foundIsect.obj_ID == chosenLightID) {
        Li_Bsdf = foundIsect.Le;
    }

    if (pdf_FF <= 0.f) {
        Lo_Bsdf = vec3(0.f);
    } else {
        Lo_Bsdf = Li_Bsdf * f_Bsdf * cosTheta_Bsdf / pdf_FF;
    }

    float pdf_FL = Pdf_Li(view_point, nor, wiW_Bsdf, chosenLightIdx);  // PDF of BSDF ray wrt light

    float w_Light = length(Lo_Light) <= 0.f ? 0.f : PowerHeuristic(1, pdf_LL, 1, pdf_LF);
    float w_Bsdf = length(Lo_Bsdf) <= 0.f ? 0.f : PowerHeuristic(1, pdf_FF, 1, pdf_FL);

    Lo = Lo_Light * w_Light + Lo_Bsdf * w_Bsdf;

    return Lo;
}

vec3 Li_Full(Ray ray)
{
    vec3 Lo = vec3(0.f);
    vec3 throughput = vec3(1.f);
    bool prev_was_specular = false;

    for (int i = 0; i < MAX_DEPTH; i++) {
        Intersection isect = sceneIntersect(ray);

        if (isect.t == INFINITY) {
            return vec3(0.f);
        }
        if (length(isect.Le) > 0.f) {
            if (i == 0 || prev_was_specular) {
                return isect.Le * throughput;
            }
            return vec3(0.f); // don't want to double count light source sampling
        }

        vec3 woW = -ray.direction;
        if (isect.material.type != SPEC_REFL && isect.material.type != SPEC_TRANS && isect.material.type != SPEC_TRANS) {
            prev_was_specular = false;
            vec3 directLight = Direct_MIS(isect, woW); // Light leaving the surface along wo that the surface recieved DIRECTLY from a light.
            Lo += directLight * throughput;
        } else {
            prev_was_specular = true;
        }

        vec2 xi = vec2(rng(), rng());

        vec3 wi_global_illum; // out;
        float pdf_gi; // PDF global illumination
        int sampledType; // out

        vec3 f = Sample_f(isect, woW, xi, wi_global_illum, pdf_gi, sampledType);
        throughput *= f * AbsDot(wi_global_illum, isect.nor) / pdf_gi;

        ray = SpawnRay(ray.origin + (ray.direction + isect.t), wi_global_illum);
    }
    return Lo;
}

void main()
{
    seed = uvec2(u_Iterations, u_Iterations + 1) * uvec2(gl_FragCoord.xy);

    Ray ray = rayCast();

    // vec3 thisIterationColor = Li_Naive(ray);
    // vec3 thisIterationColor = Li_Direct_Simple(ray);
    // vec3 thisIterationColor = Li_DirectMIS(ray);
    vec3 thisIterationColor = Li_Full(ray);

    vec3 accumulatedColor = mix(texture(u_AccumImg, fs_UV).rgb,
                                thisIterationColor,
                                1.f / float(u_Iterations));

    out_Col = vec4(accumulatedColor, 1.f);
}
 
