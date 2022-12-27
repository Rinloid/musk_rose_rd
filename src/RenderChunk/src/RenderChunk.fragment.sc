$input v_color0, v_fog, v_texcoord0, v_lightmapUV, relPos, fragPos, frameTime, waterFlag, fogControl

#include <bgfx_compute.sh>

SAMPLER2D(s_MatTexture, 0);
SAMPLER2D(s_LightMapTexture, 1);
SAMPLER2D(s_SeasonsTexture, 2);

vec3 hdrExposure(const vec3 col, const float overExposure, const float underExposure) {
    vec3 overExposed   = col / overExposure;
    vec3 normalExposed = col;
    vec3 underExposed  = col * underExposure;

    return mix(overExposed, underExposed, normalExposed);
}

/*
 ** Uncharted 2 tone mapping
 ** Link (deleted): http://filmicworlds.com/blog/filmic-tonemapping-operators/
 ** Archive: https://bit.ly/3NSGy4r
 */
vec3 uncharted2ToneMap_(vec3 x) {
    const float A = 0.015; // Shoulder strength
    const float B = 0.500; // Linear strength
    const float C = 0.100; // Linear angle
    const float D = 0.010; // Toe strength
    const float E = 0.020; // Toe numerator
    const float F = 0.300; // Toe denominator

    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}
vec3 uncharted2ToneMap(const vec3 col, const float exposureBias) {
    const float whiteLevel = 256.0;

    vec3 curr = uncharted2ToneMap_(exposureBias * col);
    vec3 whiteScale = 1.0 / uncharted2ToneMap_(vec3(whiteLevel, whiteLevel, whiteLevel));
    vec3 color = curr * whiteScale;

    return clamp(color, 0.0, 1.0);
}

vec3 contrastFilter(const vec3 col, const float contrast) {
    return (col - 0.5) * max(contrast, 0.0) + 0.5;
}

/*
 ** Atmoshpere based on one by robobo1221.
 ** See: https://www.shadertoy.com/view/Ml2cWG
*/
vec3 getAbsorption(const vec3 pos, const float posY, const float brightness) {
	vec3 absorption = pos * -posY;
	absorption = exp2(absorption) * brightness;
	
	return absorption;
}
float getRayleig(const vec3 pos, const vec3 sunPos) {
    float dist = 1.0 - clamp(distance(pos, sunPos), 0.0, 1.0);

	return 1.0 + dist * dist * 3.14;
}
float getMie(const vec3 pos, const vec3 sunPos) {
	float disk = clamp(1.0 - pow(distance(pos, sunPos), 0.1), 0.0, 1.0);
	
	return disk * disk * (3.0 - 2.0 * disk) * 2.0 * 3.14;
}
vec3 getAtmosphere(const vec3 pos, const vec3 sunPos, const vec3 skyCol, const float brightness) {
	float zenith = 0.5 / sqrt(max(pos.y, 0.05));
	
	vec3 absorption = getAbsorption(skyCol, zenith, brightness);
    vec3 sunAbsorption = getAbsorption(skyCol, 0.5 / pow(max(sunPos.y, 0.05), 0.75), brightness);
	vec3 sky = skyCol * zenith * getRayleig(pos, sunPos);
	vec3 mie = getMie(pos, sunPos) * sunAbsorption;
	
	vec3 result = mix(sky * absorption, sky / (sky + 0.5), clamp(length(max(sunPos.y, 0.0)), 0.0, 1.0));
    result += mie;
	result *= sunAbsorption * 0.5 + 0.5 * length(sunAbsorption);
	
	return result;
}

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

float fBM(vec2 x, const float amp, const float lower, const float upper, const float time, const int octaves) {
    float v = 0.0;
    float amptitude = amp;

    x += time * 0.01;

    for (int i = 0; i < octaves; i++) {
        v += amptitude * (simplexNoise(x) * 0.5 + 0.5);

        /* Optimization */
        if (v >= upper) {
            break;
        } else if (v + amptitude <= lower) {
            break;
        }

        x         *= 2.0;
        x.y       -= float(i + 1) * time * 0.025;
        amptitude *= 0.5;
    }

	return smoothstep(lower, upper, v);
}

float cloudMap(const vec2 pos, const float time, const float amp, const float rain, const int oct) {
    return fBM(pos, 0.65 - abs(amp) * 0.1, mix(0.8, 0.0, rain), 0.9, time, oct);
}

float cloudMapShade(const vec2 pos, const float time, const float amp, const float rain, const int oct) {
    return fBM(pos * 0.995, 0.64 - abs(amp) * 0.1, mix(0.8, 0.0, rain), 0.9, time, oct);
}

#define ENABLE_CLOUDS
#define ENABLE_CLOUD_SHADING

/*
 ** Generate volumetric clouds with piled 2D noise.
*/
vec2 renderClouds(const vec3 pos, const vec3 sunPos, const float brightness, const float rain, const float time) {
    const float stepSize = 0.048;
    const int cloudSteps = 5;
    const int cloudOctaves = 5;
    const int raySteps = 1;
    const float rayStepSize = 0.2;
    
    float clouds = 0.0;
    float shade = 0.0;
    float amp = -0.5;

    #ifdef ENABLE_CLOUDS
        float drawSpace = max(0.0, length(pos.xz / (pos.y * float(10))));
        if (drawSpace < 1.0 && !bool(step(pos.y, 0.0))) {
            for (int i = 0; i < cloudSteps; i++) {
                float height = 1.0 + float(i) * stepSize;
                vec2 cloudPos = pos.xz / pos.y * height;
                cloudPos *= 0.3 + hash13(floor(pos * 2048.0)) * 0.01;

                clouds = mix(clouds, 1.0, cloudMap(cloudPos, time, amp, rain, cloudOctaves));

                #ifdef ENABLE_CLOUD_SHADING
                    /* 
                    ** Compute self-casting shadows of clouds with
                    * a (sort of) volumetric ray marching!
                    */
                    vec3 rayStep = normalize(sunPos - pos) * rayStepSize;
                    vec3 rayPos = pos;
                    for (int i = 0; i < raySteps; i++) {
                        rayPos += rayStep;
                        float rayHeight = cloudMapShade(cloudPos, time, amp, rain, cloudOctaves);
                        
                        shade += mix(0.0, 1.0, max(0.0, rayHeight - (rayPos.y - pos.y)));
                    }

                #endif
                amp += 1.0 / float(cloudSteps);

            } shade /= float(cloudSteps);
        }

        clouds = mix(clouds, 0.0, drawSpace);
#   endif

    return vec2(clouds, shade);
}

vec3 getAtmosphereClouds(const vec3 pos, const vec3 sunPos, const vec3 skyCol, const float rain, const float brightness, const float daylight, const float time) {
	float zenith = 0.5 / sqrt(max(pos.y, 0.05));
	
	vec3 absorption = getAbsorption(skyCol, zenith, brightness);
    vec3 sunAbsorption = getAbsorption(skyCol, 0.5 / pow(max(sunPos.y, 0.05), 0.75), brightness);
	vec3 sky = skyCol * zenith * getRayleig(pos, sunPos);
	vec2 clouds = renderClouds(pos, sunPos, daylight, rain, time);

	vec3 mie = getMie(pos, sunPos) * sunAbsorption;
	
	vec3 result = mix(sky * absorption, sky / (sky + 0.5), clamp(length(max(sunPos.y, 0.0)), 0.0, 1.0));
	
	float cloudBrightness = clamp(dot(result, vec3(0.4, 0.4, 0.4)), 0.0, 1.0);
	vec3 cloudCol = mix(result, vec3(1.0, 1.0, 1.0), cloudBrightness);
	cloudCol = mix(cloudCol, vec3(dot(cloudCol, vec3(0.4, 0.4, 0.4)), dot(cloudCol, vec3(0.4, 0.4, 0.4)), dot(cloudCol, vec3(0.4, 0.4, 0.4))), 0.5);
	
	result = mix(result, mix(cloudCol, cloudCol * 0.6, clouds.y), 1.0 / absorption * clouds.x * 0.8);
	
    result += mie;
	result *= sunAbsorption * 0.5 + 0.5 * length(sunAbsorption);
	
	return result;
}

float getStars(const vec3 pos) {
    vec3 p = floor((normalize(pos) + 16.0) * 265.0);
    float stars = smoothstep(0.998, 1.0, hash13(p));

    return stars;
}

float getSun(const vec3 pos) {
	return 1.0 / length(pos);
}

vec3 toneMapReinhard(const vec3 color) {
	vec3 col = color * color;
    float luma = dot(col, vec3(0.4, 0.4, 0.4));
    vec3 exposure = col / (col + 1.0);
	vec3 result = mix(col / (luma + 1.0), exposure, exposure);

    return result;
}

vec3 getSky(const vec3 pos, const vec3 sunPos, const vec3 moonPos, const vec3 skyCol, const float daylight, const float rain, const float time, const int moonPhase) {
	vec3 sky = getAtmosphereClouds(pos, sunPos, skyCol, rain, mix(0.7, 2.0, smoothstep(0.0, 0.1, daylight)), daylight, time);
	sky = toneMapReinhard(sky);
	sky += mix(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0), getSun(cross(pos, sunPos) * 127.0));
	sky = mix(sky, vec3(1.0, 0.96, 0.82), getStars(pos) * smoothstep(0.4, 0.0, daylight));

	sky = mix(sky, vec3(dot(sky, vec3(0.4, 0.4, 0.4)), dot(sky, vec3(0.4, 0.4, 0.4)), dot(sky, vec3(0.4, 0.4, 0.4))), rain);
	
	return sky;
}


vec3 getSkyLight(const vec3 pos, const vec3 sunPos, const vec3 skyCol, const float daylight, const float rain) {
	vec3 sky = getAtmosphere(pos, sunPos, skyCol, mix(0.7, 2.0, smoothstep(0.0, 0.1, daylight)));
	sky = toneMapReinhard(sky);

	sky = mix(sky, vec3(dot(sky, vec3(0.4, 0.4, 0.4)), dot(sky, vec3(0.4, 0.4, 0.4)), dot(sky, vec3(0.4, 0.4, 0.4))), rain);

	return sky;
}

#define ENABLE_WATER_WAVES

/*
 ** Generate water waves with simplex noises.
*/
float getWaterWav(const vec2 pos, const float time) {
	float wav = 0.0;
    #ifdef ENABLE_WATER_WAVES
        vec2 p = pos * 0.5;

        wav += simplexNoise(vec2(p.x * 1.4 - time * 0.4, p.y + time * 0.4) * 0.6) * 4.0;
        wav += simplexNoise(vec2(p.x * 1.4 - time * 0.3, p.y + time * 0.2) * 1.2) * 1.2;
        wav += simplexNoise(vec2(p.x * 2.2 - time * 0.3, p.y * 2.8 - time * 0.6)) * 0.4;
    #endif

    return wav * 0.004;
}

/*
 ** Generate a normal map of water waves.
*/
vec3 getWaterWavNormal(const vec2 pos, const float time) {
	const float texStep = 0.04;
    
	float height = getWaterWav(pos, time);
	vec2 delta = vec2(height, height);

    delta.x -= getWaterWav(pos + vec2(texStep, 0.0), time);
    delta.y -= getWaterWav(pos + vec2(0.0, texStep), time);
    
	return normalize(vec3(delta / texStep, 1.0));
}

vec3 getTexNormal(vec2 uv, float resolution, float scale) {
    vec2 texStep = 1.0 / resolution * vec2(2.0, 1.0);
    float height = dot(texture2DLod(s_MatTexture, uv, 0.0).rgb, vec3(0.4, 0.4, 0.4));
    vec2 dxy = height - vec2(dot(texture2DLod(s_MatTexture, uv + vec2(texStep.x, 0.0), 0.0).rgb, vec3(0.4, 0.4, 0.4)),
        dot(texture2DLod(s_MatTexture, uv + vec2(0.0, texStep.y), 0.0).rgb, vec3(0.4, 0.4, 0.4)));

	return normalize(vec3(dxy * scale / texStep, 1.0));
}

mat3 getTBNMatrix(const vec3 normal) {
    vec3 T = vec3(abs(normal.y) + normal.z, 0.0, normal.x);
    vec3 B = cross(T, normal);
    vec3 N = vec3(-normal.x, normal.y, normal.z);

    return mat3(T, B, N);
}

int alpha2BlockID(const vec4 texCol) {
    bool iron   = 0.99 <= texCol.a && texCol.a < 1.00;
    bool gold   = 0.98 <= texCol.a && texCol.a < 0.99;
    bool copper = 0.97 <= texCol.a && texCol.a < 0.98;
    bool other  = 0.96 <= texCol.a && texCol.a < 0.97;

    return iron ? 0 : gold ? 1 : copper ? 2 : other ? 3 : 4;
}

/*
 * All codes below are from Origin Shader by linlin.
 * Huge thanks to their great effort.  See:
 * https://github.com/origin0110/OriginShader
*/

float getTimeFromFog(const vec4 fogCol) {
	return fogCol.g > 0.213101 ? 1.0 : 
		dot(vec4(fogCol.g * fogCol.g * fogCol.g, fogCol.g * fogCol.g, fogCol.g, 1.0), 
			vec4(349.305545, -159.858192, 30.557216, -1.628452));
}

#define SKY_COL vec3(0.4, 0.65, 1.0)
#define RAY_COL vec3(0.63, 0.62, 0.45)

#define AMBIENT_LIGHT_INTENSITY 10.0
#define SKYLIGHT_INTENSITY 30.0
#define SUNLIGHT_INTENSITY 30.0
#define MOONLIGHT_INTENSITY 10.0
#define TORCHLIGHT_INTENSITY 60.0

#define SKYLIGHT_COL vec3(0.9, 0.98, 1.0)
#define SUNLIGHT_COL vec3(1.0, 0.9, 0.85)
#define SUNLIGHT_COL_SET vec3(1.0, 0.70, 0.1)
#define TORCHLIGHT_COL vec3(1.0, 0.65, 0.3)
#define MOONLIGHT_COL vec3(0.2, 0.4, 1.0)

#define EXPOSURE_BIAS 5.0
#define GAMMA 2.3

void main() {
vec4 albedo = vec4(0.0, 0.0, 0.0, 0.0);
vec4 texCol = vec4(0.0, 0.0, 0.0, 0.0);

#if defined(DEPTH_ONLY_OPAQUE) || defined(DEPTH_ONLY)
    albedo.rgb = vec3(1.0, 1.0, 1.0);
#else
    albedo = texture2D(s_MatTexture, v_texcoord0);
    texCol = albedo;

    #if defined(ALPHA_TEST) || defined(DEPTH_ONLY)
        if (albedo.a < 0.5) {
            discard;
            return;
        }
    #endif

    #if defined(SEASONS) && (defined(OPAQUE) || defined(ALPHA_TEST))
        albedo.rgb *= mix(vec3(1.0, 1.0, 1.0), texture2D(s_SeasonsTexture, v_color0.xy).rgb * 2.0, v_color0.b);
        albedo.rgb *= v_color0.aaa;
    #else
        if (abs(v_color0.r - v_color0.g) > 0.001 || abs(v_color0.g - v_color0.b) > 0.001) {
            albedo.rgb *= normalize(v_color0.rgb);
        }
    #endif
#endif

#ifndef TRANSPARENT
    albedo.a = 1.0;
#endif

bool isMetallic = false;
#if !defined(ALPHA_TEST) && !defined(TRANSPARENT)
	if ((0.95 < texCol.a && texCol.a < 1.0) && v_color0.b == v_color0.g && v_color0.r == v_color0.g) {
		isMetallic = true;
	}
#endif

vec3 skyPos = normalize(relPos);
vec3 sunPos = vec3(-0.4, 1.0, 0.65);
float time = min(getTimeFromFog(v_fog), 0.7);
vec3 sunMoonPos = (time > 0.0 ? 1.0 : -1.0) * sunPos * vec3(cos(time), sin(time), -cos(time));
vec3 normal = normalize(cross(dFdx(fragPos), dFdy(fragPos)));
if (isMetallic) {
	normal = mul(getTexNormal(v_texcoord0, 4096.0, 0.0012), getTBNMatrix(normal));
	normal = mul(getTexNormal(v_texcoord0, 8192.0, 0.0008), getTBNMatrix(normal));
}
float outdoor = smoothstep(0.92, 0.94, v_lightmapUV.y);
float diffuse = max(0.0, dot(sunMoonPos, normal));
float daylight = max(0.0, time);
float duskDawn = min(smoothstep(0.0, 0.3, daylight), smoothstep(0.5, 0.3, daylight));
float amnientLightFactor = mix(0.2, mix(0.2, 1.4, daylight), v_lightmapUV.y);;
float dirLightFactor = mix(0.0, diffuse, outdoor);
float emissiveLightFactor = v_lightmapUV.x * v_lightmapUV.x * v_lightmapUV.x * v_lightmapUV.x * v_lightmapUV.x;
float clearWeather = 1.0 - mix(0.0, mix(smoothstep(0.5, 0.3, fogControl.x), 0.0, step(fogControl.x, 0.0)), smoothstep(0.0, 0.94, v_lightmapUV.y));
vec3 skylightCol = getSkyLight(reflect(skyPos, normal), sunMoonPos, SKY_COL, daylight, 1.0 - clearWeather);
vec3 sunlightCol = mix(SUNLIGHT_COL, SUNLIGHT_COL_SET, duskDawn);
vec3 daylightCol = mix(skylightCol, sunlightCol, 0.4);
vec3 ambientLightCol = mix(mix(vec3(0.0, 0.0, 0.0), TORCHLIGHT_COL, emissiveLightFactor), mix(MOONLIGHT_COL, daylightCol, daylight), dirLightFactor);
ambientLightCol += 1.0 - max(max(ambientLightCol.r, ambientLightCol.g), ambientLightCol.b);
float vanillaAO = 0.0;
#ifndef SEASONS
	vanillaAO = 1.0 - (v_color0.g * 2.0 - (v_color0.r < v_color0.b ? v_color0.r : v_color0.b)) * 1.4;
#endif
float occlShadow = mix(1.0, 0.2, vanillaAO);

vec3 light = vec3(0.0, 0.0, 0.0);

light += ambientLightCol * AMBIENT_LIGHT_INTENSITY * amnientLightFactor * occlShadow;
light += sunlightCol * SUNLIGHT_INTENSITY * dirLightFactor * daylight * clearWeather;
light += MOONLIGHT_COL * MOONLIGHT_INTENSITY * dirLightFactor * (1.0 - daylight) * clearWeather;
light += skylightCol * SKYLIGHT_INTENSITY * dirLightFactor * daylight * clearWeather;
light += TORCHLIGHT_COL * TORCHLIGHT_INTENSITY * emissiveLightFactor;

albedo.rgb = pow(albedo.rgb, vec3(GAMMA, GAMMA, GAMMA));
albedo.rgb *= light;
albedo.rgb = hdrExposure(albedo.rgb, EXPOSURE_BIAS, 0.2);
albedo.rgb = uncharted2ToneMap(albedo.rgb, EXPOSURE_BIAS);
albedo.rgb = pow(albedo.rgb, vec3(1.0 / GAMMA, 1.0 / GAMMA, 1.0 / GAMMA));
albedo.rgb = contrastFilter(albedo.rgb, GAMMA - 0.6);

if (waterFlag > 0.5 || isMetallic) {
	if (waterFlag > 0.5) {
		normal = mul(getWaterWavNormal(fragPos.xz, frameTime), getTBNMatrix(normalize(cross(dFdx(fragPos), dFdy(fragPos)))));
	}

	float cosTheta = 1.0 - abs(dot(normalize(relPos), normal));
	vec3 sky = getSky(reflect(skyPos, normal), sunMoonPos, sunMoonPos, SKY_COL, daylight, 1.0 - clearWeather, frameTime, 7);

	if (waterFlag > 0.5) {
		albedo.rgb = mix(albedo.rgb, v_color0.rgb, outdoor) * 0.5;
		albedo.a = mix(0.2, 1.0, cosTheta);
	}
	
	albedo.rgb = mix(albedo.rgb, sky, cosTheta * outdoor);

	float specularLight = getSun(cross(reflect(skyPos, normal), sunMoonPos) * 45.0);
	albedo += specularLight * outdoor;
}

float fogBrightness = mix(0.7, 2.0, smoothstep(0.0, 0.1, daylight));
vec3 fogCol = toneMapReinhard(getAtmosphere(skyPos, sunMoonPos, SKY_COL, fogBrightness));

albedo.rgb = mix(albedo.rgb, mix(fogCol, vec3(dot(fogCol, vec3(0.4, 0.4, 0.4)), dot(fogCol, vec3(0.4, 0.4, 0.4)), dot(fogCol, vec3(0.4, 0.4, 0.4))), 1.0 - clearWeather), v_fog.a);

#if !defined(TRANSPARENT)
	if (!isMetallic) {
		float sunRayFactor = !bool(step(fogControl.x, 0.0)) ? min(smoothstep(0.5, 0.94, v_lightmapUV.y) * max(0.0, 1.0 - distance(skyPos, sunMoonPos)) * smoothstep(0.0, 0.1, daylight), 1.0) : 0.0;
		albedo.rgb = mix(albedo.rgb, RAY_COL, sunRayFactor);
	}
#endif

    gl_FragColor = albedo;
}
