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
