$input v_fog, relPos, frameTime, fogControl

#include <bgfx_shader.sh>

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

float getBlockyClouds(const vec2 pos, const float frameTime, const float rainLevel) {
    vec2 p = pos * 0.5;
    p += frameTime * 0.02;
    float body = hash12(floor(p));
    body = body > mix(0.7, 0.0, rainLevel) ? 1.0 : 0.0;

    return body;
}

vec2 getClouds(const vec3 pos, const vec3 sunPos, const float frameTime, const float rainLevel) {
    const int cloudSteps = 32;
    const float cloudStepSize = 0.006;
    const int raySteps = 16;
    const float rayStepSize = 0.04;
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

#define EARTH_RADIUS 6372000.0
#define ATMOSPHERE_RADIUS 6472000.0
#define RAYLEIGH_SCATTERING_COEFFICIENT vec3(0.000055, 0.00013, 0.000224)
#define MIE_SCATTERING_COEFFICIENT vec3(0.00002, 0.00002, 0.00002)
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

    vec3 rayOrig = vec3(0.0, EARTH_RADIUS + (ATMOSPHERE_RADIUS - EARTH_RADIUS) * 0.3, 0.0);

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

    totalSky = getAtmosphere(pos, shadowLightPos, frameTime, rainLevel, mix(2.0, 20.0, daylight));
    totalSky = mix(totalSky, vec3(getLuma(totalSky), getLuma(totalSky), getLuma(totalSky)), rainLevel);
    totalSky *= mix(0.65, 1.0, bayerX64(screenPos) * 0.5 + 0.5);

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

void main() {
vec4 albedo = vec4(0.0, 0.0, 0.0, 1.0);

float time = 1.0;
vec3 sunPos = normalize(vec3(cos(time), sin(time), 0.2));
vec3 moonPos = -sunPos;
vec3 shadowLightPos = time > 0.0 ? sunPos : moonPos;
vec2 screenPos = gl_FragCoord.xy;
float daylight = max(0.0, time);
float rainLevel = 0.0;

albedo.rgb = getSky(normalize(relPos), sunPos, moonPos, shadowLightPos, screenPos, daylight, frameTime, rainLevel);

    gl_FragColor = albedo;
}
