$input v_fog, relPos, frameTime, fogColtrol

#include <bgfx_shader.sh>

vec3 getAbsorption(const vec3 pos, const float posY, const float brightness) {
	vec3 absorption = pos * -posY;
	absorption = exp2(absorption) * brightness;
	
	return absorption;
}

float getSunSpot(const vec3 pos, const vec3 sunPos) {
	return smoothstep(0.03, 0.025, distance(pos, sunPos)) * 25.0;
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
	float zenith = 0.5 / pow(max(pos.y, 0.05), 0.5);
	
	vec3 absorption = getAbsorption(skyCol, zenith, brightness);
    vec3 sunAbsorption = getAbsorption(skyCol, 0.5 / pow(max(sunPos.y, 0.05), 0.75), brightness);
	vec3 sky = skyCol * zenith * getRayleig(pos, sunPos);
	vec3 sun = getSunSpot(pos, sunPos) * absorption;
	vec3 mie = getMie(pos, sunPos) * sunAbsorption;
	
	vec3 result = mix(sky * absorption, sky / (sky + 0.5), clamp(length(max(sunPos.y, 0.0)), 0.0, 1.0));
    result += sun + mie;
	result *= sunAbsorption * 0.5 + 0.5 * length(sunAbsorption);
	
	return result;
}

vec3 toneMapReinhard(const vec3 col){
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    vec3 exposure = col / (col + 1.0);

    return mix(col / (luma + 1.0), exposure, exposure);
}

/*
 ** Uncharted 2 tone mapping
 ** Link (deleted): http://filmicworlds.com/blog/filmic-tonemapping-operators/
 ** Archive: https://bit.ly/3NSGy4r
 */
vec3 uncharted2ToneMap_(vec3 x) {
    const float A = 0.15; // Shoulder strength
    const float B = 0.50; // Linear strength
    const float C = 0.10; // Linear angle
    const float D = 0.10; // Toe strength
    const float E = 0.02; // Toe numerator
    const float F = 0.30; // Toe denominator

    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}
vec3 uncharted2ToneMap(vec3 frag, float exposureBias) {
    const float whiteLevel = 11.2;

    vec3 curr = uncharted2ToneMap_(exposureBias * frag);
    vec3 whiteScale = 1.0 / uncharted2ToneMap_(vec3(whiteLevel, whiteLevel, whiteLevel));
    vec3 color = curr * whiteScale;

    return clamp(color, 0.0, 1.0);
}

vec3 contrastFilter(vec3 color, float contrast) {
    float t = 0.5 - contrast * 0.5;

    return clamp(color * contrast + t, 0.0, 1.0);
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
    float snoise(vec2 v) {
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
    float snoise(vec2 p) {
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

        return dot(n, vec3(70.0, 70.0, 70.0)) * 1.5;
    }
#endif

float fBM(vec2 x, const float amp, const float lower, const float upper, const float time, const int octaves) {
    float v = 0.0;
    float amptitude = amp;

    x += time * 0.02;

    for (int i = 0; i < octaves; i++) {
        v += amptitude * (snoise(x) * 0.5 + 0.5);

        if (v >= upper) {
            break;
        } else if (v + amptitude <= lower) {
            break;
        }

        x         *= 2.0;
        x.y       -= float(i + 1) * time * 0.04;
        amptitude *= 0.5;
    }

	return smoothstep(lower, upper, v);
}

vec2 renderClouds(const vec3 pos, const vec3 sunMoonPos, const float rain, const float time) {
    const int steps = 16;
    const float stepSize = 0.04;

	vec2 clouds = vec2(0.0, 0.0);
	float amp = mix(0.25, 1.0, rain);

	for (int i = 0; i < steps; i++) {
		float height = 1.0 + float(i) * stepSize;
		vec2 cloudPos = pos.xz / pos.y * height;
		cloudPos *= 0.3;
		clouds.x += fBM(cloudPos, amp, 0.4, 0.8, time, 5);
		if (clouds.x > 0.0) {
			clouds.y += fBM(cloudPos * 0.9, amp, 0.4, 1.0, time, 5);
		}
		amp *= 1.07;
	} clouds /= float(steps);

	clouds.x = smoothstep(0.0, mix(0.25, 0.5, rain), clouds.x);
    clouds.y = smoothstep(0.0, mix(0.50, 1.0, rain), clouds.y);

	return clouds;
}

float getStars(const vec3 pos) {
    vec3 p = floor((normalize(pos) + 16.0) * 265.0);
    float stars = smoothstep(0.998, 1.0, hash13(p));

    return stars;
}

float drawSun(const vec3 pos) {
	return inversesqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
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

float getTimeFromLightmap(const float daylight) {
    return (((39.581994 * daylight - 74.236058) * daylight + 33.842220) * daylight + 9.368113) * daylight - 7.363836;
}
bool equ3(const vec3 v) {
	return abs(v.x - v.y) < 0.000002 && abs(v.y - v.z) < 0.000002;
}
bool isUnderwater(const vec3 normal, const vec2 uv1, const vec3 worldPos, const vec4 texCol, const vec4 vertexCol) {
	return normal.y > 0.9
	    && uv1.y < 0.9
	    && abs((2.0 * worldPos.y - 15.0) / 16.0 - uv1.y) < 0.00002
	    && !equ3(texCol.rgb)
	    && (equ3(vertexCol.rgb) || vertexCol.a < 0.00001)
	    && abs(fract(worldPos.y) - 0.5) > 0.00001;
}

#define SUN_MOON_DIR vec3(0.4, 0.65, 1.0)

void main() {
vec3 albedo;

float time = getTimeFromFog(v_fog);
float rain = smoothstep(0.5, 0.3, fogColtrol.x);
vec3 sunMoonPos = (time > 0.0 ? 1.0 : -1.0) * SUN_MOON_DIR * vec3(cos(time), sin(time), -cos(time));
float daylight = max(0.0, time);

float skyBrightness = mix(1.0, 2.0, smoothstep(0.0, 0.1, daylight));
vec3 skyCol = getAtmosphere(normalize(relPos), sunMoonPos, vec3(0.4, 0.65, 1.0), skyBrightness);

albedo = skyCol;
albedo = toneMapReinhard(albedo);
albedo = contrastFilter(albedo, 1.8);

float drawSpace = max(0.0, length(relPos.xz / (relPos.y * float(20))));
if (drawSpace < 1.0 && !bool(step(relPos.y, 0.0))) {
	float clouds = renderClouds(relPos, sunMoonPos, rain, frameTime).x;
	float shade = renderClouds(relPos, sunMoonPos, rain, frameTime).y;

    float cloudBrightness = clamp(dot(skyCol, vec3(0.2126, 0.7152, 0.0722)) * 2.0, 0.0, 1.0);
    vec3 cloudCol = vec3(cloudBrightness, cloudBrightness, cloudBrightness);
	cloudCol *= mix(1.0, 0.5, shade);

	albedo = mix(albedo, cloudCol, mix(clouds * 0.5, 0.0, drawSpace));
    albedo = mix(albedo, vec3(1.0, 1.0, 1.0), getStars(normalize(relPos) * 0.5 * (1.0 - daylight)));
    albedo = mix(albedo, vec3(1.0, 1.0, 1.0), drawSun(cross(normalize(relPos), sunMoonPos) * 25.0) * (1.0 - drawSpace));
}

    gl_FragColor = vec4(albedo, 1.0);
}