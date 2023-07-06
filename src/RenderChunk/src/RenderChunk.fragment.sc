$input v_color0, v_fog, v_texcoord0, v_lightmapUV, relPos, fragPos, frameTime, waterFlag, fogControl

#include <bgfx_compute.sh>

SAMPLER2D(s_MatTexture, 0);
SAMPLER2D(s_SeasonsTexture, 1);

float bayerX2(vec2 a) {
    return fract(dot(floor(a), vec2(0.5, floor(a).y * 0.75)));
}

#define bayerX4(a)  (bayerX2 (0.5 * (a)) * 0.25 + bayerX2(a))
#define bayerX8(a)  (bayerX4 (0.5 * (a)) * 0.25 + bayerX2(a))
#define bayerX16(a) (bayerX8 (0.5 * (a)) * 0.25 + bayerX2(a))
#define bayerX32(a) (bayerX16(0.5 * (a)) * 0.25 + bayerX2(a))
#define bayerX64(a) (bayerX32(0.5 * (a)) * 0.25 + bayerX2(a))

/** Hash from "Hahs without Sine"
 ** Author: David Hoskins
 ** https://www.shadertoy.com/view/4djSRW
*/

float hash11(float p) {
    p = fract(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}
float hash12(vec2 p) {
	vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);

    return fract((p3.x + p3.y) * p3.z);
}
float hash13(vec3 p3) {
	p3  = fract(p3 * .1031);
    p3 += dot(p3, p3.zyx + 31.32);

    return fract((p3.x + p3.y) * p3.z);
}
vec2 hash22(vec2 p) {
    p = vec2(dot(p,vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

/*
 ** Generate ripples based on Ripple Splatter by AntoninHS.
 ** See: https://www.shadertoy.com/view/fdfSzs
*/
vec2 scale(const vec2 pos, const vec2 pivotPoint, const float scale, const float minScale, const float maxScale){
    vec2 p = pos - pivotPoint;
    p *= 1.0 / (max(mix(minScale, maxScale, scale), 0.01));
    p += pivotPoint;

    return p;
}
vec3 offset(const vec2 pos, const vec2 intPos, const vec2 offset, const float time){
    vec2 p = intPos + offset;
    float rand13 = hash12(p + vec2(25.0, 48.0));
    vec2 timeOffsetRand = vec2(hash11(floor(rand13 + time)), hash11(floor(rand13 + time))) + p;
    float rand11 = hash12(timeOffsetRand);
    float rand12 = hash12(timeOffsetRand + vec2(17.0, 33.0));
    vec2 rand21 = vec2(rand11, rand12);
    
    vec2 newPos = pos - offset;
    newPos -= rand21;
    
    float newScale = cos(time + rand11 * 6.28318530718) * 0.5 + 0.5;
    newPos = scale(newPos, vec2(0.5, 0.5), 0.0, 1.0, newScale);
    
    vec3 randPos = vec3(newPos, rand13);

    return randPos;
}
float getRippleSplatter(const vec3 randPos, const float time){
    vec2 p = randPos.xy;
    p += vec2(0.5, 0.5);
    p += clamp(p, vec2(-0.5, -0.5), vec2(0.5, 0.5));

    float cone = 1.0 - distance(p, vec2(0.0, 0.0));
    
    float cycleTime = fract(randPos.z + time);
    float animatedCone = cone + cycleTime;
    float rippleArea = clamp(animatedCone, 0.0, 1.0) - clamp((animatedCone -0.5) * 1.5, 0.0, 1.0);
    float activateRipples = sin(cycleTime * 3.14159265359);
    
    float result = animatedCone * 18.0;
    result = cos(result);
    result *= rippleArea * activateRipples;
    result = clamp(result, 0.0, 1.0);
    
    float randMask = floor(randPos.z + time) + randPos.z * 512.0;
    randMask = step(hash12(vec2(randMask, randMask)), 1.0);

    result *= cone * randMask;
    result = clamp(result, 0.0, 1.0);
    
    return result;
}
float getRipples(const vec2 fragPos, const float time) {
    vec2 p = fragPos * 2.5;
    
    vec3 offset00 = offset(p - floor(p), floor(p), vec2(0.0, 0.0), time);
    vec3 offset01 = offset(p - floor(p), floor(p), vec2(0.0, 1.0), time);
    vec3 offset10 = offset(p - floor(p), floor(p), vec2(1.0, 0.0), time);
    vec3 offset11 = offset(p - floor(p), floor(p), vec2(1.0, 1.0), time);
    
    
    float ripple00 = getRippleSplatter(offset00, time);
    float ripple01 = getRippleSplatter(offset01, time);
    float ripple10 = getRippleSplatter(offset10, time);
    float ripple11 = getRippleSplatter(offset11, time);
    
    float ripples = max(max(ripple00, ripple01), max(ripple10, ripple11));

    return ripples;
}

float getRainRipples(const vec3 worldNormal, const vec2 fragPos, const float rainLevel, const float time) {
    float result = 0.0;

    if (rainLevel > 0.0) {
        result = getRipples(fragPos, time) * mix(0.0, max(0.0, worldNormal.y), rainLevel);
    }

    return result;
}

vec3 getRainRipplesNormal(const vec3 worldNormal, const vec2 fragPos, const float rainLevel, const float time) {
	const float texStep = 5.0;
    
	float height = getRainRipples(worldNormal, fragPos, rainLevel, time);
	vec2  delta  = vec2(height, height);

    delta.x -= getRainRipples(worldNormal, fragPos + vec2(texStep, 0.0), rainLevel, time);
    delta.y -= getRainRipples(worldNormal, fragPos + vec2(0.0, texStep), rainLevel, time);
    
	return normalize(vec3(delta / texStep, 1.0));
}

#ifdef USE_ASHIMA
    /*
     ** Simplex Noise modified by Rin
     ** Original author: Ashima Arts (MIT License)
     ** See: https://github.com/ashima/webgl-noise
     **      https://github.com/stegu/webgl-noise
    */
    vec2 mod289(vec2 x) {
        return x - floor(x * 1.0 / 289.0) * 289.0;
    }
    vec3 mod289(vec3 x) {
        return x - floor(x * 1.0 / 289.0) * 289.0;
    }
    vec3 permute289(vec3 x) {
        return mod289((x * 34.0 + 1.0) * x);
    }
    float simplexNoise(vec2 v) {
        const vec4 C = vec4(
            0.211324865405187,   // (3.0-sqrt(3.0))/6.0
            0.366025403784439,   // 0.5*(sqrt(3.0)-1.0)
           -0.577350269189626,   // -1.0 + 2.0 * C.x
            0.024390243902439);  // 1.0 / 41.0

        // First corner
        vec2 i  = floor(v + dot(v, C.yy));
        vec2 x0 = v -   i + dot(i, C.xx);

        // Other corners
        vec2 i1  = x0.x > x0.y ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
        vec4 x12 = x0.xyxy + C.xxzz;
        x12.xy -= i1;

        // Permutations
        i = mod289(i); // Avoid truncation effects in permutation
        vec3 p =
            permute289(
                permute289(
                    i.y + vec3(0.0, i1.y, 1.0)
                    ) + i.x + vec3(0.0, i1.x, 1.0)
                );

        vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
        m = m * m;
        m = m * m;

        // Gradients: 41 points uniformly over a line, mapped onto a
        // diamond.  The ring size 17*17 = 289 is close to a multiple of
        // 41 (41*7 = 287)
        vec3 x  = 2.0 * fract(p * C.www) - 1.0;
        vec3 h  = abs(x) - 0.5;
        vec3 ox = round(x);
        vec3 a0 = x - ox;

        // Normalise gradients implicitly by scaling m
        m *= inversesqrt(a0 * a0 + h * h);

        // Compute final noise value at P
        vec3 g;
        g.x  = a0.x  * x0.x   + h.x  * x0.y;
        g.yz = a0.yz * x12.xz + h.yz * x12.yw;
        return 130.0 * dot(m, g);
    }
#else
    /*
     ** Simplex Noise modified by Rin
     ** Original author: Inigo Quilez (MIT License)
     ** See: https://www.shadertoy.com/view/Msf3WH
    */
    float simplexNoise(vec2 p) {
        const float K1 = 0.366025403784439; // (sqrt(3) - 1) / 2;
        const float K2 = 0.211324865405187; // (3 - sqrt(3)) / 6;

        vec2  i = floor(p + (p.x + p.y) * K1);
        vec2  a = p - i + (i.x + i.y) * K2;
        float m = step(a.y, a.x); 
        vec2  o = vec2(m, 1.0 - m);
        vec2  b = a - o + K2;
        vec2  c = a - 1.0 + 2.0 * K2;
        vec3  h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
        vec3  n = h * h * h * h;
        
        n *= vec3(dot(a, hash22(i + 0.0)), dot(b, hash22(i + o)), dot(c, hash22(i + 1.0)));

        return dot(n, vec3(130.0, 130.0, 130.0));
    }
#endif

float getWaterWaves(const vec2 pos, const float frameTime) {
	float waves = 0.0;
    vec2 p = pos * 0.5;

    waves += simplexNoise(vec2(p.x * 1.4 - frameTime * 0.4, p.y + frameTime * 0.4) * 0.6) * 4.0;
    waves += simplexNoise(vec2(p.x * 1.4 - frameTime * 0.3, p.y + frameTime * 0.2) * 1.2) * 1.2;
    waves += simplexNoise(vec2(p.x * 2.2 - frameTime * 0.3, p.y * 2.8 - frameTime * 0.6)) * 0.4;

	return waves * 0.004;
}

vec3 getWaterWaveNormal(const vec2 pos, const float frameTime) {
	const float texStep = 0.04;
    
	float height = getWaterWaves(pos, frameTime);
	vec2  delta  = vec2(height, height);

    delta.x -= getWaterWaves(pos + vec2(texStep, 0.0), frameTime);
    delta.y -= getWaterWaves(pos + vec2(0.0, texStep), frameTime);
    
	return normalize(vec3(delta / texStep, 1.0));
}

vec3 getTexNormal(const vec2 uv, const float resolution, const float scale) {
    vec2 texStep = 1.0 / resolution * vec2(2.0, 1.0);

    float height = dot(texture2DLod(s_MatTexture, uv, 0.0).rgb, vec3(0.22, 0.707, 0.0714));
    vec2  delta  =  vec2(height, height);

    delta.x -= dot(texture2DLod(s_MatTexture, uv + vec2(texStep.x, 0.0), 0.0).rgb, vec3(0.22, 0.707, 0.071));
    delta.y -= dot(texture2DLod(s_MatTexture, uv + vec2(0.0, texStep.y), 0.0).rgb, vec3(0.22, 0.707, 0.071));

	return normalize(vec3(delta * scale / texStep, 1.0));
}

mat3 getTBNMatrix(const vec3 normal) {
    vec3 T = vec3(abs(normal.y) + normal.z, 0.0, normal.x);
    vec3 B = vec3(0.0, -abs(normal).x - abs(normal).z, abs(normal).y);
    vec3 N = normal;

    return transpose(mat3(T, B, N));
}

float getBlockyClouds(const vec2 pos, const float frameTime, const float rainLevel) {
    vec2 p = pos * 0.5;
    p += frameTime * 0.02;
    float body = hash12(floor(p));
    body = body > mix(0.7, 0.0, rainLevel) ? 1.0 : 0.0;

    return body;
}

vec2 getClouds(const vec3 pos, const vec3 sunPos, const float frameTime, const float rainLevel) {
    const int cloudSteps = 8;
    const float cloudStepSize = 0.024;
    const int raySteps = 6;
    const float rayStepSize = 0.32;
    const float cloudHeight = 256.0;

    float clouds = 0.0;
    float shade  = 0.0;
    float highlight = 0.0;

    float amp = 0.5;

    float drawSpace = max(0.0, length(pos.xz / (pos.y * float(16))));
    if (drawSpace < 1.0 && !bool(step(pos.y, 0.0))) {
        for (int i = 0; i < cloudSteps; i++) {
            float height = 1.0 + float(i) * cloudStepSize;
            vec3 p = pos.xyz / pos.y * height;
            clouds += getBlockyClouds(p.xz * 4.0, frameTime, rainLevel);

            if (clouds > 0.0) {
                vec3 rayPos = pos.xyz / pos.y * height;
                float ray = 0.0;
                for (int j = 0; j < raySteps; j++) {
                    ray += getBlockyClouds(rayPos.xz * 4.0, frameTime, rainLevel) * 0.6;
                    rayPos += sunPos * rayStepSize;
                } ray /= float(raySteps);
                shade += ray;
            }
            amp -= 0.2 / float(cloudSteps);
        } clouds /= float(cloudSteps);
          shade /= float(cloudSteps);

        clouds = mix(clouds, 0.0, drawSpace);
    }

    return vec2(clouds, shade);
}

#define EARTH_RADIUS 637200.0
#define ATMOSPHERE_RADIUS 647200.0
#define EARTH_RADIUS_THE_END 6372000.0
#define ATMOSPHERE_RADIUS_THE_END 6472000.0
#define RAYLEIGH_SCATTERING_COEFFICIENT vec3(0.000055, 0.00013, 0.000224)
#define MIE_SCATTERING_COEFFICIENT vec3(0.00004, 0.00004, 0.00004)
#define RAYLEIGH_SCALE_HEIGHT 8000.0
#define MIE_SCALE_HEIGHT 1200.0
#define MIE_DIRECTION 0.75

float getLuma(const vec3 col) {
    return dot(col, vec3(0.22, 0.707, 0.071));
}

vec3 tonemapReinhard(const vec3 col) {
    return clamp(1.0 - exp(-1.0 * col), 0.0, 1.0);
}

vec2 getRaySphereIntersection(const vec3 rayDir, const vec3 rayOrig, const float raySphere) {
    float PoD = dot(rayOrig, rayDir);
    float raySphereSquared = raySphere * raySphere;

    float delta = PoD * PoD + raySphereSquared - dot(rayOrig, rayOrig);
    if (delta < 0.0) return vec2(-1.0, -1.0);
    delta = sqrt(delta);

    return -PoD + vec2(-delta, delta);
}

vec3 getAtmosphere(const vec3 pos, const vec3 sunPos, const float frameTime, const float rainLevel, const float intensity) {
    const int numSteps = 32;

    vec3 totalSky = vec3(0.0, 0.0, 0.0);

    vec3 rayOrig = vec3(0.0, EARTH_RADIUS, 0.0);

    vec2 p = getRaySphereIntersection(pos, rayOrig, ATMOSPHERE_RADIUS);
    p.y = min(p.y, getRaySphereIntersection(pos, rayOrig, EARTH_RADIUS).x);

    float rayStepsize = (p.y - p.x) / float(numSteps);
    float raySteps = 0.0;

    vec3 totalRayleigh = vec3(0.0, 0.0, 0.0);
    vec3 totalMie = vec3(0.0, 0.0, 0.0);

    float rayLeighOpticalDepth = 0.0;
    float mieOpticalDepth = 0.0;
    
    float rayleighPhase = 3.0 / (16.0 * 3.14159265359) * (1.0 + dot(pos, sunPos) * dot(pos, sunPos));
    float miePhase = 3.0 / (8.0 * 3.14159265359) * ((1.0 - MIE_DIRECTION * MIE_DIRECTION) * (dot(pos, sunPos) * dot(pos, sunPos) + 1.0)) / (pow(1.0 + MIE_DIRECTION * MIE_DIRECTION - 2.0 * dot(pos, sunPos) * MIE_DIRECTION, 1.5) * (2.0 + MIE_DIRECTION * MIE_DIRECTION));

    for (int i = 0; i < numSteps; i++) {
        vec3 rayPos = rayOrig + pos * (raySteps + rayStepsize * 0.5);
        float rayHeight = length(rayPos) - EARTH_RADIUS;

        rayLeighOpticalDepth += exp(-rayHeight / RAYLEIGH_SCALE_HEIGHT) * rayStepsize;
        mieOpticalDepth += exp(-rayHeight / MIE_SCALE_HEIGHT) * rayStepsize;

        totalRayleigh += exp(-rayHeight / RAYLEIGH_SCALE_HEIGHT) * rayStepsize * exp(-(MIE_SCATTERING_COEFFICIENT * mieOpticalDepth + RAYLEIGH_SCATTERING_COEFFICIENT * rayLeighOpticalDepth));
        totalMie += exp(-rayHeight / MIE_SCALE_HEIGHT) * rayStepsize * exp(-(MIE_SCATTERING_COEFFICIENT * mieOpticalDepth + RAYLEIGH_SCATTERING_COEFFICIENT * rayLeighOpticalDepth));

        raySteps += rayStepsize;
    }
    totalSky = intensity * (rayleighPhase * RAYLEIGH_SCATTERING_COEFFICIENT * totalRayleigh + miePhase * MIE_SCATTERING_COEFFICIENT * totalMie);

    return tonemapReinhard(totalSky);
}

vec3 getAtmosphereClouds(const vec3 pos, const vec3 sunPos, const float frameTime, const float rainLevel, const float intensity) {
    const int numSteps = 32;

    vec3 totalSky = vec3(0.0, 0.0, 0.0);

    vec3 rayOrig = vec3(0.0, EARTH_RADIUS, 0.0);

    vec2 p = getRaySphereIntersection(pos, rayOrig, ATMOSPHERE_RADIUS);
    p.y = min(p.y, getRaySphereIntersection(pos, rayOrig, EARTH_RADIUS).x);

    float rayStepsize = (p.y - p.x) / float(numSteps);
    float raySteps = 0.0;

    vec3 totalRayleigh = vec3(0.0, 0.0, 0.0);
    vec3 totalMie = vec3(0.0, 0.0, 0.0);

    float rayLeighOpticalDepth = 0.0;
    float mieOpticalDepth = 0.0;
    
    float rayleighPhase = 3.0 / (16.0 * 3.14159265359) * (1.0 + dot(pos, sunPos) * dot(pos, sunPos));
    float miePhase = 3.0 / (8.0 * 3.14159265359) * ((1.0 - MIE_DIRECTION * MIE_DIRECTION) * (dot(pos, sunPos) * dot(pos, sunPos) + 1.0)) / (pow(1.0 + MIE_DIRECTION * MIE_DIRECTION - 2.0 * dot(pos, sunPos) * MIE_DIRECTION, 1.5) * (2.0 + MIE_DIRECTION * MIE_DIRECTION));

    for (int i = 0; i < numSteps; i++) {
        vec3 rayPos = rayOrig + pos * (raySteps + rayStepsize * 0.5);
        float rayHeight = length(rayPos) - EARTH_RADIUS;

        rayLeighOpticalDepth += exp(-rayHeight / RAYLEIGH_SCALE_HEIGHT) * rayStepsize;
        mieOpticalDepth += exp(-rayHeight / MIE_SCALE_HEIGHT) * rayStepsize;

        totalRayleigh += exp(-rayHeight / RAYLEIGH_SCALE_HEIGHT) * rayStepsize * exp(-(MIE_SCATTERING_COEFFICIENT * mieOpticalDepth + RAYLEIGH_SCATTERING_COEFFICIENT * rayLeighOpticalDepth));
        totalMie += exp(-rayHeight / MIE_SCALE_HEIGHT) * rayStepsize * exp(-(MIE_SCATTERING_COEFFICIENT * mieOpticalDepth + RAYLEIGH_SCATTERING_COEFFICIENT * rayLeighOpticalDepth));

        raySteps += rayStepsize;
    }
    totalSky = intensity * (rayleighPhase * RAYLEIGH_SCATTERING_COEFFICIENT * totalRayleigh + miePhase * MIE_SCATTERING_COEFFICIENT * totalMie);

    vec2 clouds = getClouds(pos, sunPos, frameTime, rainLevel);
    totalSky = intensity * 50.0 * miePhase * MIE_SCATTERING_COEFFICIENT * totalMie * clouds.x * exp(-clouds.y * 5.0) + totalSky * exp(-clouds.x);

    return tonemapReinhard(totalSky);
}

vec3 getAtmosphereTheEnd(const vec3 pos, const vec3 sunPos, const float frameTime, const float rainLevel, const float intensity) {
    const int numSteps = 32;

    vec3 totalSky = vec3(0.0, 0.0, 0.0);

    vec3 rayOrig = vec3(0.0, EARTH_RADIUS_THE_END + (ATMOSPHERE_RADIUS_THE_END - EARTH_RADIUS_THE_END) * 0.3, 0.0);

    vec2 p = getRaySphereIntersection(pos, rayOrig, ATMOSPHERE_RADIUS_THE_END);
    p.y = min(p.y, getRaySphereIntersection(pos, rayOrig, EARTH_RADIUS_THE_END).x);

    float rayStepsize = (p.y - p.x) / float(numSteps);
    float raySteps = 0.0;

    vec3 totalRayleigh = vec3(0.0, 0.0, 0.0);
    vec3 totalMie = vec3(0.0, 0.0, 0.0);

    float rayLeighOpticalDepth = 0.0;
    float mieOpticalDepth = 0.0;
    
    float rayleighPhase = 3.0 / (16.0 * 3.14159265359) * (1.0 + dot(pos, sunPos) * dot(pos, sunPos));
    float miePhase = 3.0 / (8.0 * 3.14159265359) * ((1.0 - MIE_DIRECTION * MIE_DIRECTION) * (dot(pos, sunPos) * dot(pos, sunPos) + 1.0)) / (pow(1.0 + MIE_DIRECTION * MIE_DIRECTION - 2.0 * dot(pos, sunPos) * MIE_DIRECTION, 1.5) * (2.0 + MIE_DIRECTION * MIE_DIRECTION));

    for (int i = 0; i < numSteps; i++) {
        vec3 rayPos = rayOrig + pos * (raySteps + rayStepsize * 0.5);
        float rayHeight = length(rayPos) - EARTH_RADIUS_THE_END;

        rayLeighOpticalDepth += exp(-rayHeight / RAYLEIGH_SCALE_HEIGHT) * rayStepsize;
        mieOpticalDepth += exp(-rayHeight / MIE_SCALE_HEIGHT) * rayStepsize;

        totalRayleigh += exp(-rayHeight / RAYLEIGH_SCALE_HEIGHT) * rayStepsize * exp(-(MIE_SCATTERING_COEFFICIENT * mieOpticalDepth + RAYLEIGH_SCATTERING_COEFFICIENT * rayLeighOpticalDepth));
        totalMie += exp(-rayHeight / MIE_SCALE_HEIGHT) * rayStepsize * exp(-(MIE_SCATTERING_COEFFICIENT * mieOpticalDepth + RAYLEIGH_SCATTERING_COEFFICIENT * rayLeighOpticalDepth));

        raySteps += rayStepsize;
    }
    totalSky = intensity * (rayleighPhase * RAYLEIGH_SCATTERING_COEFFICIENT * totalRayleigh + miePhase * MIE_SCATTERING_COEFFICIENT * totalMie);

//    vec2 clouds = getClouds(pos, sunPos, frameTime, rainLevel);
//    totalSky = intensity * 25.0 * miePhase * MIE_SCATTERING_COEFFICIENT * totalMie * clouds.x * exp(-clouds.y * 5.0) + totalSky * exp(-clouds.x);

    return tonemapReinhard(totalSky);
}

float getSun(const vec3 pos, const vec3 sunPos) {
	return smoothstep(0.05, 0.035, distance(pos, sunPos));
}

float getMoon(const vec3 pos, const vec3 moonPos) {
	return smoothstep(0.07, 0.05, distance(pos, moonPos));
}

float getStars(const vec3 pos, const float time) {
    vec3 p = floor((pos + time * 0.001) * 265.0);
    float stars = smoothstep(0.998, 1.0, hash13(p));

    return stars;
}

vec3 getSky(const vec3 pos, const vec3 sunPos, const vec3 moonPos, const vec3 shadowLightPos, const vec2 screenPos, const float daylight, const float frameTime, const float rainLevel) {
    vec3 totalSky = vec3(0.0, 0.0, 0.0);

    totalSky = getAtmosphereClouds(pos * mix(0.65, 1.0, bayerX64(screenPos.xy) * 0.5 + 0.5), shadowLightPos, frameTime, rainLevel, mix(2.0, 20.0, daylight));
    totalSky = mix(totalSky, vec3(getLuma(totalSky), getLuma(totalSky), getLuma(totalSky)), rainLevel);
    totalSky *= mix(0.65, 1.0, bayerX64(screenPos.xy) * 0.5 + 0.5);

    totalSky = mix(totalSky, vec3(1.0, 1.0, 1.0), getStars(pos, frameTime) * (1.0 - daylight) * (1.0 - rainLevel));
    totalSky = mix(totalSky, vec3(1.0, 1.0, 1.0), getSun(pos, sunPos) * (1.0 - rainLevel));
    totalSky = mix(totalSky, vec3(1.0, 0.95, 0.81), getMoon(pos, moonPos) * (1.0 - rainLevel));

    return clamp(totalSky, 0.0, 1.0);
}

vec3 getSkyTheEnd(const vec3 pos, const vec3 sunPos, const vec3 moonPos, const vec3 shadowLightPos, const vec2 screenPos, const float daylight, const float frameTime, const float rainLevel) {
    vec3 totalSky = vec3(0.0, 0.0, 0.0);

    totalSky = getAtmosphereTheEnd(pos, shadowLightPos, frameTime, rainLevel, mix(2.0, 20.0, daylight));
    totalSky = mix(totalSky, vec3(getLuma(totalSky), getLuma(totalSky), getLuma(totalSky)), rainLevel);

    float drawSpace = max(0.0, length(pos.xz / (pos.y * float(16))));
    if (drawSpace < 1.0 && !bool(step(pos.y, 0.0))) {
        totalSky = mix(totalSky, vec3(1.0, 1.0, 1.0), getStars(pos, frameTime) * (1.0 - drawSpace));
    }
    totalSky = mix(totalSky, vec3(1.0, 1.0, 1.0), getSun(pos, sunPos));

    return totalSky;
}

float getTime(const vec4 fogCol) {
	return fogCol.g > 0.213101 ? 1.0 : 
		dot(vec4(fogCol.g * fogCol.g * fogCol.g, fogCol.g * fogCol.g, fogCol.g, 1.0), 
			vec4(349.305545, -159.858192, 30.557216, -1.628452));
}

vec3 brighten(const vec3 col) {
    float rgbMax = max(col.r, max(col.g, col.b));
    float delta  = 1.0 - rgbMax;

    return col + delta;
}

#define pointlightCol vec3(1.00, 0.66, 0.28)
#define moonlightCol vec3(1.0, 0.98, 0.8)

vec3 getSunlightCol(const float daylight) {
    const vec3 setCol = vec3(1.00, 0.36, 0.02);
    const vec3 dayCol = vec3(1.00, 0.87, 0.80);

    return mix(setCol, dayCol, max(0.75, daylight));
}

vec3 getSkylightCol(const vec3 pos, const vec3 sunPos, const vec3 moonPos, const vec3 shadowLightPos, const float daylight, const float frameTime, const float rainLevel) {
    vec3 totalSky = vec3(0.0, 0.0, 0.0);

    totalSky = getAtmosphere(pos, shadowLightPos, frameTime, rainLevel, mix(2.0, 20.0, daylight));
    totalSky = mix(totalSky, vec3(getLuma(totalSky), getLuma(totalSky), getLuma(totalSky)), rainLevel);

    return clamp(totalSky, 0.0, 1.0);
}

vec3 getSkylightColTheEnd(const vec3 pos, const vec3 sunPos, const vec3 moonPos, const vec3 shadowLightPos, const float daylight, const float frameTime, const float rainLevel) {
    vec3 totalSky = vec3(0.0, 0.0, 0.0);

    totalSky = getAtmosphereTheEnd(pos, shadowLightPos, frameTime, rainLevel, mix(2.0, 20.0, daylight));
    totalSky = mix(totalSky, vec3(getLuma(totalSky), getLuma(totalSky), getLuma(totalSky)), rainLevel);

    return totalSky;
}

vec3 getAmbientLightCol(const vec3 pos, const vec3 sunPos, const vec3 moonPos, const vec3 shadowLightPos, const float daylight, const float frameTime, const float rainLevel, const float moonHeight, const float outdoor, const float pointLightLevel) {
    vec3 totalAmbientLightCol = vec3(0.0, 0.0, 0.0);

    totalAmbientLightCol = mix(mix(vec3(1.0, 1.0, 1.0), pointlightCol, pointLightLevel), mix(moonlightCol, mix(getSunlightCol(daylight), getSkylightCol(pos, sunPos, moonPos, shadowLightPos, daylight, frameTime, rainLevel), 0.5), daylight), outdoor);

    return totalAmbientLightCol;
}

#define AMBIENTLIGHT_INTENSITY 10.0
#define POINTLIGHT_INTENSITY 180.0
#define SUNLIGHT_INTENSITY 50.0
#define SKYLIGHT_INTENSITY 50.0
#define MOONLIGHT_INTENSITY 10.0

#define RAIN_CUTOFF 0.5

#define GAMMA 2.2

vec3 getAmbientLight(const vec3 pos, const vec3 sunPos, const vec3 moonPos, const vec3 shadowLightPos, const float daylight, const float frameTime, const float rainLevel, const float moonHeight, const float outdoor, const float pointLightLevel) {
    vec3 totalAmbientLight = vec3(0.0, 0.0, 0.0);
    
    float intensity = AMBIENTLIGHT_INTENSITY;
    totalAmbientLight = intensity * getAmbientLightCol(pos, sunPos, moonPos, shadowLightPos, daylight, frameTime, rainLevel, moonHeight, outdoor, pointLightLevel);

    return totalAmbientLight;
}

vec3 getSunlight(const float daylight, const vec4 directionalShadowCol, const float rainLevel) {
    vec3 totalSunLight = vec3(0.0, 0.0, 0.0);

    float intensity = SUNLIGHT_INTENSITY * daylight;
    intensity *= mix(1.0, RAIN_CUTOFF, rainLevel);

    vec3 shadowCutoff = intensity * (1.0 - directionalShadowCol.rgb);
    totalSunLight = intensity * getSunlightCol(daylight) + (intensity * getSunlightCol(daylight) - shadowCutoff);
    totalSunLight = mix(totalSunLight, vec3(0.0, 0.0, 0.0), directionalShadowCol.a);

    return totalSunLight;
}

vec3 getMoonlight(const float moonHeight, const vec4 directionalShadowCol, const float rainLevel) {
    vec3 totalMoonLight = vec3(0.0, 0.0, 0.0);

    float intensity = MOONLIGHT_INTENSITY * moonHeight;
    intensity *= mix(1.0, RAIN_CUTOFF, rainLevel);

    vec3 shadowCutoff = intensity * (1.0 - directionalShadowCol.rgb);
    totalMoonLight = intensity * moonlightCol + (intensity * moonlightCol - shadowCutoff);
    totalMoonLight = mix(totalMoonLight, vec3(0.0, 0.0, 0.0), directionalShadowCol.a);

    return totalMoonLight;
}

vec3 getSkylight(const float outdoor, const vec3 pos, const vec3 sunPos, const vec3 moonPos, const vec3 shadowLightPos, const float daylight, const float frameTime, const float rainLevel) {
    vec3 totalSkyLight = vec3(0.0, 0.0, 0.0);

    float intensity = SKYLIGHT_INTENSITY;
    intensity *= mix(1.0, RAIN_CUTOFF, rainLevel);

    totalSkyLight = intensity * getSkylightCol(pos, sunPos, moonPos, shadowLightPos, daylight, frameTime, rainLevel);
    totalSkyLight *= outdoor;

    return totalSkyLight;
}

vec3 getSkylightTheEnd(const float outdoor, const vec3 pos, const vec3 sunPos, const vec3 moonPos, const vec3 shadowLightPos, const float daylight, const float frameTime, const float rainLevel) {
    vec3 totalSkyLight = vec3(0.0, 0.0, 0.0);

    float intensity = SKYLIGHT_INTENSITY;
    intensity *= mix(1.0, RAIN_CUTOFF, rainLevel);

    totalSkyLight = intensity * getSkylightColTheEnd(pos, sunPos, moonPos, shadowLightPos, daylight, frameTime, rainLevel);
    totalSkyLight *= outdoor;

    return totalSkyLight;
}

vec3 getPointLight(const float pointLightLevel, const float outdoor, const float daylight, const float rainLevel) {
    vec3 totalPointLight = vec3(0.0, 0.0, 0.0);

    float intensity = POINTLIGHT_INTENSITY * pointLightLevel;
    intensity *= mix(mix(1.0, 0.0, smoothstep(0.7, 0.94, outdoor * daylight)), RAIN_CUTOFF, rainLevel);

    totalPointLight = intensity * pointlightCol;

    return totalPointLight;
}

float fresnelSchlick(const vec3 H, const vec3 N, const float reflectance) {
    float cosTheta = clamp(1.0 - max(0.0, dot(H, N)), 0.0, 1.0);

    return clamp(reflectance + (1.0 - reflectance) * cosTheta * cosTheta * cosTheta * cosTheta * cosTheta, 0.0, 1.0);
}

vec3 fresnelSchlick(const vec3 H, const vec3 N, const vec3 reflectance) {
    float cosTheta = clamp(1.0 - max(0.0, dot(H, N)), 0.0, 1.0);

    return clamp(reflectance + (1.0 - reflectance) * cosTheta * cosTheta * cosTheta * cosTheta * cosTheta, 0.0, 1.0);
}

vec3 getPBRSpecular(const vec3 V, const vec3 L, const vec3 N, const float R, const vec3 reflectance) {
    vec3  H = normalize(V + L);
    float D = (R * R)
            / (3.14159265359 * (max(0.0, dot(H, N)) * max(0.0, dot(H, N)) * (R * R - 1.0) + 1.0) * (max(0.0, dot(H, N)) * max(0.0, dot(H, N)) * (R * R - 1.0) + 1.0));
    float G = ((max(0.0, dot(V, N))) / (max(0.0, dot(V, N)) + ((R + 1.0) * (R + 1.0)) * 0.125))
            * ((max(0.0, dot(L, N))) / (max(0.0, dot(L, N)) + ((R + 1.0) * (R + 1.0)) * 0.125));
    vec3  F = fresnelSchlick(H, V, reflectance);

    return vec3(clamp((D * G * F) / max(0.001, 4.0 * max(dot(N, V), 0.0) * max(dot(L, N), 0.0)), 0.0, 1.0));
}

// https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
vec3 getEnvironmentBRDF(const vec3 V, const vec3 N, const float R, const vec3 reflectance) {
    vec4 r = R * vec4(-1.0, -0.0275, -0.572,  0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
    vec2 AB = vec2(-1.04, 1.04) * min(r.x * r.x, exp2(-9.28 * max(0.0, dot(V, N)))) * r.x + r.y + r.zw;

    return reflectance * AB.x + AB.y;
}

/* 
 ** Uncharted 2 tonemapping
 ** See: http://filmicworlds.com/blog/filmic-tonemapping-operators/
*/
vec3 uncharted2TonemapFilter(const vec3 col) {
    const float A = 0.015; // Shoulder strength
    const float B = 0.500; // Linear strength
    const float C = 0.100; // Linear angle
    const float D = 0.010; // Toe strength
    const float E = 0.020; // Toe numerator
    const float F = 0.300; // Toe denominator

    return ((col * (A * col + C * B) + D * E) / (col * (A * col + B) + D * F)) - E / F;
}
vec3 uncharted2Tonemap(const vec3 col, const float whiteLevel, const float exposure) {
    vec3 curr = uncharted2TonemapFilter(col * exposure);
    vec3 whiteScale = 1.0 / uncharted2TonemapFilter(vec3(whiteLevel, whiteLevel, whiteLevel));
    vec3 color = curr * whiteScale;

    return color;
}

vec3 contrastFilter(const vec3 col, const float contrast) {
    return (col - 0.5) * max(contrast, 0.0) + 0.5;
}

vec3 hdrExposure(const vec3 col, const float over, const float under) {
    return mix(col / over, col * under, col);
}

int getBlockID(const vec4 texCol) {
    bool iron     = 0.99 <= texCol.a && texCol.a < 1.00;
    bool gold     = 0.98 <= texCol.a && texCol.a < 0.99;
    bool copper   = 0.97 <= texCol.a && texCol.a < 0.98;
    bool other    = 0.96 <= texCol.a && texCol.a < 0.97;

    return iron ? 0 : gold ? 1 : copper ? 2 : other ? 3 : 4;
}

void main() {
vec4 albedo = vec4(0.0, 0.0, 0.0, 0.0);
vec4 texCol = vec4(0.0, 0.0, 0.0, 0.0);

#if defined(DEPTH_ONLY_OPAQUE) || defined(DEPTH_ONLY)
        albedo.rgb = vec3(1.0, 1.0, 1.0);
#   else
        albedo = texture2D(s_MatTexture, v_texcoord0);
        texCol = albedo;

#   ifdef ALPHA_TEST
        if (albedo.a < 0.5) discard;
#   endif

#   if defined(SEASONS) && (defined(OPAQUE) || defined(ALPHA_TEST))
        albedo.rgb *= mix(vec3(1.0, 1.0, 1.0), texture2D(s_SeasonsTexture, v_color0.xy).rgb * 2.0, v_color0.b);
        albedo.rgb *= v_color0.aaa;
#   else
        if (abs(v_color0.r - v_color0.g) > 0.001 || abs(v_color0.g - v_color0.b) > 0.001) albedo.rgb *= normalize(v_color0.rgb);
#   endif
#endif

#ifndef TRANSPARENT
    albedo.a = 1.0;
#endif

/* Hacks */
bool isUnderwater = fogControl.x == 0.0;
bool isNether = bool(step(0.1, fogControl.x / fogControl.y) - step(0.12, fogControl.x / fogControl.y));
bool isTheEnd = (v_fog.r > v_fog.g && v_fog.b > v_fog.g)
    && (greaterThan(v_fog.rgb, vec3(0.03, 0.02, 0.04)) == bvec3(true, true, true))
    && (lessThan(v_fog.rgb, vec3(0.05, 0.04, 0.06)) == bvec3(true, true, true));

vec3 pos = normalize(relPos);
vec2 screenPos = gl_FragCoord.xy;
vec3 viewDir = -pos;

float time = isTheEnd ? 1.0 : getTime(v_fog);
vec3 sunPos = normalize(vec3(cos(time), sin(time), 0.2));
vec3 moonPos = -sunPos;
vec3 shadowLightPos = time > 0.0 ? sunPos : moonPos;
float daylight = max(0.0, time);
float moonHeight = max(0.0, sin(moonPos.y));
float rainLevel = isTheEnd ? 0.0 : mix(0.0, mix(smoothstep(0.5, 0.3, fogControl.x), 0.0, step(fogControl.x, 0.0)), smoothstep(0.0, 0.94, v_lightmapUV.y));
float vanillaAO = 0.0;
#ifndef SEASONS
	vanillaAO = 1.0 - (v_color0.g * 2.0 - (v_color0.r < v_color0.b ? v_color0.r : v_color0.b));
#endif

vec3 normal = normalize(cross(dFdx(fragPos), dFdy(fragPos)));
if (waterFlag > 0.5) {
    albedo.rgb = mix(albedo.rgb, vec3(0.02, 0.1, 0.2), v_lightmapUV.y);
    texCol.rgb = albedo.rgb;
    normal = normalize(mul(getWaterWaveNormal(fragPos.xz, frameTime), getTBNMatrix(normal)));
} else {
    normal = normalize(mul(getTexNormal(v_texcoord0, 1024.0, 0.0005), getTBNMatrix(normal)));
#   if !defined(ALPHA_TEST) && !defined(TRANSPARENT)
        if ((0.95 < texCol.a && texCol.a < 1.0) && v_color0.r == v_color0.g && v_color0.g == v_color0.b) {
            if (getBlockID(texCol) == 0) { // Iron
                normal = normalize(mul(getTexNormal(v_texcoord0, 8192.0, 0.0006), getTBNMatrix(normalize(cross(dFdx(fragPos), dFdy(fragPos))))));
            } else if (getBlockID(texCol) == 1) { // Gold
                normal = normalize(mul(getTexNormal(v_texcoord0, 4096.0, 0.0005), getTBNMatrix(normalize(cross(dFdx(fragPos), dFdy(fragPos))))));
            } else if (getBlockID(texCol) == 2) { // Copper
                normal = normalize(mul(getTexNormal(v_texcoord0, 2048.0, 0.0005), getTBNMatrix(normalize(cross(dFdx(fragPos), dFdy(fragPos))))));
            } else if (getBlockID(texCol) == 3) { // Others
                normal = normalize(mul(getTexNormal(v_texcoord0, 4096.0, 0.0005), getTBNMatrix(normalize(cross(dFdx(fragPos), dFdy(fragPos))))));
            }
        }
#   endif
}
if (rainLevel > 0.0) {
    normal = normalize(mul(getRainRipplesNormal(normal, fragPos.xz, rainLevel, frameTime), getTBNMatrix(normal)));
}

bool isReflective = false;

float roughness = 0.6;
float F0 = 0.5;
#if !defined(ALPHA_TEST) && !defined(TRANSPARENT)
	if ((0.95 < texCol.a && texCol.a < 1.0) && v_color0.r == v_color0.g && v_color0.g == v_color0.b) {
        if (getBlockID(texCol) == 0) { // Iron
            isReflective = true;
            roughness = 0.09;
            F0 = 0.72;
        } else if (getBlockID(texCol) == 1) { // Gold
            isReflective = true;
            roughness = 0.12;
            F0 = 0.68;
        } else if (getBlockID(texCol) == 2) { // Copper
            isReflective = true;
            roughness = 0.24;
            F0 = 0.52;
        } else if (getBlockID(texCol) == 3) { // Others
            isReflective = true;
            roughness = 0.27;
            F0 = 0.3;
        }
    }
#endif
vec3 reflectance = mix(vec3(0.04, 0.04, 0.04), texCol.rgb, F0);
if (waterFlag > 0.5) {
    roughness = 0.02;
    F0 = 0.6;
    reflectance = vec3(0.6, 0.6, 0.6);
} else if (rainLevel > 0.0 && v_lightmapUV.y > 0.9375) {
    float rainRipples = mix(0.0, getRainRipples(normalize(cross(dFdx(fragPos), dFdy(fragPos))), fragPos.xz, rainLevel, frameTime), smoothstep(0.90, 0.94, v_lightmapUV.y));
    roughness = mix(roughness, 0.02, rainRipples);
    F0 = mix(F0, 0.9, rainRipples);
}

vec4 directionalShadowCol = vec4(0.0, 0.0, 0.0, 0.0);
float outdoor = isTheEnd ? 1.0 : v_lightmapUV.y;
float shadow = isTheEnd ? 0.0 : smoothstep(0.94, 0.92, outdoor);
float diffuse = max(0.0, dot(shadowLightPos, normal));
directionalShadowCol.a = mix(1.0, shadow, diffuse);

float pointLightLevel = v_lightmapUV.x * v_lightmapUV.x * v_lightmapUV.x * v_lightmapUV.x * v_lightmapUV.x;

vec3 directionalLight = vec3(0.0, 0.0, 0.0);
vec3 undirectionalLight = vec3(0.0, 0.0, 0.0);

undirectionalLight += getAmbientLight(pos, sunPos, moonPos, shadowLightPos, daylight, frameTime, rainLevel, moonHeight, outdoor, pointLightLevel) * (1.0 - vanillaAO);
directionalLight   += getSunlight(daylight, directionalShadowCol, rainLevel);
directionalLight   += getMoonlight(moonHeight, directionalShadowCol, rainLevel);
undirectionalLight += isTheEnd ? getSkylight(outdoor, pos, sunPos, moonPos, shadowLightPos, daylight, frameTime, rainLevel) * (1.0 - vanillaAO) : getSkylight(outdoor, pos, sunPos, moonPos, shadowLightPos, daylight, frameTime, rainLevel) * (1.0 - vanillaAO);
undirectionalLight += getPointLight(pointLightLevel, outdoor, daylight, rainLevel) * (1.0 - vanillaAO);

vec3 totalLight = undirectionalLight + directionalLight;
totalLight *= mix(0.65, 1.0, bayerX64(gl_FragCoord.xy) * 0.5 + 0.5);

albedo.rgb *= totalLight;

albedo.rgb = uncharted2Tonemap(albedo.rgb, 112.0, 1.25);
albedo.rgb = hdrExposure(albedo.rgb, 112.0, 1.25);

vec3 specular = getPBRSpecular(viewDir, shadowLightPos, normal, roughness, reflectance);
vec3 fresnel = fresnelSchlick(viewDir, normal, reflectance);
vec3 directionalLightRatio = max(vec3(0.2, 0.2, 0.2) * outdoor, directionalLight / max(vec3(0.001, 0.001, 0.001), totalLight));
vec3 reflection = mix(albedo.rgb, isTheEnd ? getSkyTheEnd(reflect(pos, normal), sunPos, moonPos, shadowLightPos, screenPos, daylight, frameTime, rainLevel) : getSky(reflect(pos, normal), sunPos, moonPos, shadowLightPos, screenPos, daylight, frameTime, rainLevel), outdoor);
reflection *= getEnvironmentBRDF(viewDir, normal, roughness, reflectance);

if (waterFlag > 0.5) {
    albedo.rgb *= 1.0 - reflectance;
    albedo.rgb += (reflection * fresnel) + (10.0 * directionalLightRatio * specular);
    albedo.a = mix(mix(0.1, 1.0, fresnelSchlick(viewDir, normal, F0)), 1.0, 10.0 * directionalLightRatio.r * specular.r);
} else if (isReflective) {
    albedo.rgb *= 1.0 - reflectance;
    albedo.rgb += (reflection * fresnel) + (mix(3.5, 5.0, rainLevel) * directionalLightRatio * specular);
} else {
    albedo.rgb += (mix(2.5, 5.0, rainLevel) * directionalLightRatio * specular);
}

vec3 fogCol = getSkylightCol(pos, sunPos, moonPos, shadowLightPos, daylight, frameTime, rainLevel);
fogCol = mix(vec3(getLuma(fogCol), getLuma(fogCol), getLuma(fogCol)), fogCol, outdoor);
if (isTheEnd) fogCol = getSkylightColTheEnd(pos, sunPos, moonPos, shadowLightPos, daylight, frameTime, rainLevel);
if (isUnderwater) fogCol = v_fog.rgb; 
fogCol *= mix(0.65, 1.0, bayerX64(gl_FragCoord.xy) * 0.5 + 0.5);

albedo.rgb = mix(albedo.rgb, fogCol, v_fog.a);

    gl_FragColor = albedo;
}