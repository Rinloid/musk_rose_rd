$input v_fog, relPos, frameTime, fogControl

#include <bgfx_compute.sh>

#define ENABLE_SHADER_SKY

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

float getBlockyClouds(const vec2 pos, const float time, const float rainLevel) {
    vec2 p = pos * 0.5;
    p += time * 0.02;
    float body = hash12(floor(p));
    body = body > mix(0.7, 0.0, rainLevel) ? 1.0 : 0.0;

    return body;
}

vec3 getClouds(const vec3 pos, const vec3 lightPos, const float time, const float rainLevel) {
    const int cloudSteps = 16;
    const float cloudStepSize = 0.012;
    const int raySteps = 8;
    const float rayStepSize = 0.16;
    const float cloudHeight = 256.0;

    float clouds = 0.0;
    float shade  = 0.0;
    float highlight = 0.0;

    float amp = 0.5;

    float drawSpace = max(0.0, length(pos.xz / (pos.y * float(8))));
    if (drawSpace < 1.0 && !bool(step(pos.y, 0.0))) {
        for (int i = 0; i < cloudSteps; i++) {
            float height = 1.0 + float(i) * cloudStepSize;
            vec3 p = pos.xyz / pos.y * height;
            clouds += getBlockyClouds(p.xz * 2.0, time, rainLevel);

            if (clouds > 0.0) {
                vec3 rayPos = pos.xyz / pos.y * height;
                float ray = 0.0;
                for (int j = 0; j < raySteps; j++) {
                    ray += getBlockyClouds(rayPos.xz * 2.0, time, rainLevel) * 0.6;
                    rayPos += lightPos * rayStepSize;
                } ray /= float(raySteps);
                shade += ray;
                highlight += ray * ray * ray * ray * ray * ray;
            }
            amp -= 0.2 / float(cloudSteps);
        } clouds /= float(cloudSteps);
          shade /= float(cloudSteps);
          highlight /= float(cloudSteps);

        clouds = mix(clouds, 0.0, drawSpace);
    }

    return vec3(clouds, shade, 1.0 - highlight);
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
float getSunPoint(const vec3 pos, const vec3 sunPos, const float rainLevel) {
	return smoothstep(0.1, 0.0, distance(pos, sunPos)) * 5.0 * (1.0 - rainLevel);
}
float getRayleig(const vec3 pos, const vec3 sunPos) {
    float dist = 1.0 - clamp(distance(pos, sunPos), 0.0, 1.0);

	return 1.0 + dist * dist * 3.14;
}
float getMie(const vec3 pos, const vec3 sunPos) {
	float disk = clamp(1.0 - pow(distance(pos, sunPos), 0.1), 0.0, 1.0);
	
	return disk * disk * (3.0 - 2.0 * disk) * 2.0 * 3.14;
}
vec3 getAtmosphere(const vec3 pos, const vec3 sunPos, const vec3 skyCol, const vec3 vanillaSkyCol, const vec3 vanillaFogCol, const float brightness) {
	vec3 result = mix(vanillaSkyCol, vanillaFogCol, smoothstep(0.8, 1.0, 1.0 - pos.y));
	#ifdef ENABLE_SHADER_SKY
		float zenith = 0.5 / sqrt(max(pos.y, 0.05));
		
		vec3 absorption = getAbsorption(skyCol, zenith, brightness);
		vec3 sunAbsorption = getAbsorption(skyCol, 1.0 / pow(max(sunPos.y, 0.05), 0.75), brightness);
		vec3 sky = skyCol * zenith * getRayleig(pos, sunPos);
		vec3 mie = getMie(pos, sunPos) * sunAbsorption;
		
		result = mix(sky * absorption, sky / (sky + 0.5), clamp(length(max(sunPos.y, 0.0)), 0.0, 1.0));
		result += mie;
		result *= sunAbsorption * 0.5 + 0.5 * length(sunAbsorption);
	#endif

	return result;
}

vec3 getAtmosphereClouds(const vec3 pos, const vec3 sunPos, const vec3 skyCol, const vec3 vanillaSkyCol, const vec3 vanillaFogCol, const float brightness, const float daylight, const float rainLevel, const float time) {
	vec3 result = mix(vanillaSkyCol, vanillaFogCol, smoothstep(0.8, 1.0, 1.0 - pos.y));
	#ifdef ENABLE_SHADER_SKY
		const float cloudDensity = 1.50; // [0.00 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00 1.10 1.20 1.30 1.40 1.50 1.60 1.70 1.80 1.90 2.00 2.10 2.20 2.30 2.40 2.50 2.60 2.70 2.80 2.90 3.00 3.10 3.20 3.30 3.40 3.50 3.60 3.70 3.80 3.90 4.00 4.10 4.20 4.30 4.40 4.50 4.60 4.70 4.80 4.90 5.00 5.10 5.20 5.30 5.40 5.50 5.60 5.70 5.80 5.90 6.00 6.10 6.20 6.30 6.40 6.50 6.60 6.70 6.80 6.90 7.00 7.10 7.20 7.30 7.40 7.50 7.60 7.70 7.80 7.90 8.00 8.10 8.20 8.30 8.40 8.50 8.60 8.70 8.80 8.90 9.00 9.10 9.20 9.30 9.40 9.50 9.60 9.70 9.80 9.90 10.00]
		
		float zenith = 0.5 / sqrt(max(pos.y, 0.05));
		
		vec3 absorption = getAbsorption(skyCol, zenith, brightness);
		vec3 sunAbsorption = getAbsorption(skyCol, 1.0 / pow(max(sunPos.y, 0.05), 0.75), brightness);
		vec3 sky = skyCol * zenith * getRayleig(pos, sunPos);
		vec3 sun = getSunPoint(pos, sunPos, rainLevel) * absorption;
		vec3 clouds = getClouds(pos, sunPos, time, rainLevel);

		vec3 mie = getMie(pos, sunPos) * sunAbsorption;
		
		result = mix(sky * absorption, sky / (sky + 0.5), clamp(length(max(sunPos.y, 0.0)), 0.0, 1.0));
		
		float cloudBrightness = clamp(dot(result, vec3(0.22, 0.707, 0.071)), 0.0, 1.0);
		vec3 cloudCol = mix(result, vec3(1.0, 1.0, 1.0), cloudBrightness);
		cloudCol = mix(cloudCol, vec3(dot(cloudCol, vec3(0.22, 0.707, 0.071)), dot(cloudCol, vec3(0.22, 0.707, 0.071)), dot(cloudCol, vec3(0.22, 0.707, 0.071))), 0.4);
		
		result = sun + mix(result, mix(cloudCol * mix(1.0, 1.5, clouds.z), cloudCol * mix(0.0, 0.65, daylight), clouds.y), 1.0 / absorption * clouds.x * cloudDensity);
		
		result += mie;
		result *= sunAbsorption * 0.5 + 0.5 * length(sunAbsorption);
	#endif

	return result;
}

float getLuma(const vec3 col) {
	return dot(col, vec3(0.22, 0.707, 0.071));
}

vec3 toneMapReinhard(const vec3 color) {
	vec3 col = color * color;
    float luma = getLuma(col);
    vec3 exposure = col / (col + 1.0);
	vec3 result = mix(col / (luma + 1.0), exposure, exposure);

    return result;
}

float getStars(const vec3 pos, const float time) {
    vec3 p = floor((pos + time * 0.001) * 265.0);
    float stars = smoothstep(0.998, 1.0, hash13(p));

    return stars;
}

float getMoonPhase(const int phase) {
	/*
	 ** moonPhase variable: [0, 1, 2, 3, 4, 5, 6, 7]
	 ** Moon in MC:         [4, 5, 6, 7, 0, 1, 2, 3]
	 ** 0 = new moon; 7 = full moon.
	*/
	int correctedPhase = 0 <= phase && phase < 5 ? phase + 4 : phase;

	// [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
	return float(correctedPhase) * 0.25 * 3.14159265359;
}

float diffuseSphere(const vec3 spherePos, const float radius, const vec3 lightPos) {
    float sq = radius * radius - spherePos.x * spherePos.x - spherePos.y * spherePos.y - spherePos.z * spherePos.z;

    if (sq < 0.0) {
        return 0.0;
    } else {
        float z = sqrt(sq);
        vec3 normal = normalize(vec3(spherePos.yx, z));
		
        return max(0.0, dot(normal, lightPos));
    }
}

vec4 getMoon(const vec3 moonPosition, const float moonPhase, const float moonScale) {
	vec3 lightPos = vec3(sin(moonPhase), 0.0, -cos(moonPhase));
    float m = diffuseSphere(moonPosition, moonScale, lightPos);
	float moonTex = mix(1.0, 0.85, clamp(simplexNoise(moonPosition.xz * 0.2), 0.0, 1.0));
	m = smoothstep(0.0, 0.3, m) * moonTex;
    
	return vec4(mix(vec3(0.1, 0.05, 0.01), vec3(1.0, 0.95, 0.81), m), diffuseSphere(moonPosition, moonScale, vec3(0.0, 0.0, 1.0)));
}

vec3 getSky(const vec3 pos, const vec3 sunPos, const vec3 skyCol, const vec3 vanillaSkyCol, const vec3 vanillaFogCol, const float daylight, const float rainLevel, const float time, const int moonPhase) {
	vec3 sky = getAtmosphereClouds(pos, sunPos, skyCol, vanillaSkyCol, vanillaFogCol, mix(0.5, 2.0, smoothstep(0.0, 0.1, daylight)), daylight, rainLevel, time);
	vec4 moon = getMoon(cross(pos, -sunPos) * 127.0, getMoonPhase(moonPhase), 7.0);
	moon.a = mix(moon.a, 0.0, rainLevel);

	sky = toneMapReinhard(sky);
    sky = mix(sky, vec3(1.0, 1.0, 1.0), (1.0 / length(cross(pos, sunPos) * 20.0)) * daylight);
	sky = mix(sky, vec3(1.0, 1.0, 1.0), getStars(pos, time) * (1.0 - smoothstep(0.0, 0.3, daylight)) * (1.0 - rainLevel));
	sky = mix(sky, moon.rgb, moon.a * smoothstep(0.1, 0.0, daylight));
	sky = mix(sky, vec3(getLuma(sky), getLuma(sky), getLuma(sky)), rainLevel);

	return sky;
}

float getTimeFromFog(const vec4 fogCol) {
	return fogCol.g > 0.213101 ? 1.0 : 
		dot(vec4(fogCol.g * fogCol.g * fogCol.g, fogCol.g * fogCol.g, fogCol.g, 1.0), 
			vec4(349.305545, -159.858192, 30.557216, -1.628452));
}

void main() {
vec3 albedo = vec3(0.0, 0.0, 0.0);
float time = getTimeFromFog(v_fog);
vec3 shadowLightPos = vec3(-0.4, 0.8, 1.0) * vec3(cos(time), sin(time), -cos(time));
float daylight = max(0.0, sin(time) * 0.8);
float rainLevel = mix(smoothstep(0.5, 0.3, fogControl.x), 0.0, step(fogControl.x, 0.0));

albedo = getSky(normalize(relPos), shadowLightPos, vec3(0.4, 0.65, 1.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), daylight, rainLevel, frameTime, 0);

    gl_FragColor = vec4(albedo, 1.0);
}