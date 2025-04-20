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
