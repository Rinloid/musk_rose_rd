$input v_color0, v_fog, v_texcoord0, v_lightmapUV, relPos, chunkPos, frameTime, waterFlag, fogColtrol

#include <bgfx_shader.sh>

SAMPLER2D(s_MatTexture, 0);
SAMPLER2D(s_LightMapTexture, 1);
SAMPLER2D(s_SeasonsTexture, 2);

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

float getLuma(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 toneMapReinhard(const vec3 col){
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    vec3 exposure = col / (col + 1.0);

    return mix(col / (luma + 1.0), exposure, exposure);
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
    const int steps = 4;
    const float stepSize = 0.16;

	vec2 clouds = vec2(0.0, 0.0);
	float amp = mix(0.25, 1.0, rain);

	for (int i = 0; i < steps; i++) {
		float height = 1.0 + float(i) * stepSize;
		vec2 cloudPos = pos.xz / pos.y * height + hash13(floor(pos * 1024.0)) * 0.02;
		cloudPos *= 0.3;
		clouds.x += fBM(cloudPos, amp, 0.4, 0.8, time, 4);
		if (clouds.x > 0.0) {
			clouds.y += fBM(cloudPos * 0.9, amp, 0.4, 1.0, time, 3);
		}
		amp *= 1.28;
	} clouds /= float(steps);

	clouds.x = smoothstep(0.0, mix(0.25, 0.5, rain), clouds.x);
    clouds.y = smoothstep(0.0, mix(0.50, 1.0, rain), clouds.y);

	return clouds;
}

mat3 getTBNMatrix(const vec3 normal) {
    vec3 T = vec3(abs(normal.y) + normal.z, 0.0, normal.x);
    vec3 B = cross(T, normal);
    vec3 N = vec3(-normal.x, normal.y, normal.z);
    return mat3(T, B, N);
}

float getWaterWaves(vec2 p, const float time) {
	float r = 0.0;

    r += snoise(vec2(p.x * 1.4 - time * 0.4, p.y * 0.65 + time * 0.4) * 0.6) * 3.0;
    r += snoise(vec2(p.x * 1.0 + time * 0.6, p.y - time * 0.75)) * 0.5;
    r += snoise(vec2(p.x * 2.2 - time * 0.3, p.y * 2.8 - time * 0.6)) * 0.25;

	return r * 0.005;
}

vec3 getWaterWavesNormal(const vec2 pos, const float time) {
	const float texStep = 0.04;
	float height = getWaterWaves(pos, time);
	vec2 dxy = height - vec2(getWaterWaves(pos + vec2(texStep, 0.0), time),
		getWaterWaves(pos + vec2(0.0, texStep), time));
    
	return normalize(vec3(dxy / texStep, 1.0));
}

int alpha2BlockID(const vec4 texCol) {
    bool iron   = 0.99 <= texCol.a && texCol.a < 1.00;
    bool gold   = 0.98 <= texCol.a && texCol.a < 0.99;
    bool copper = 0.97 <= texCol.a && texCol.a < 0.98;
    bool other  = 0.96 <= texCol.a && texCol.a < 0.97;

    return iron ? 0 : gold ? 1 : copper ? 2 : other ? 3 : 4;
}

vec3 getF0(const vec4 texCol, const vec4 albedo) {
    mat4 f0 =
        mat4(vec4(0.56, 0.57, 0.58, 1.0), // Iron
             vec4(1.00, 0.71, 0.29, 1.0), // Gold
             vec4(0.95, 0.64, 0.54, 1.0), // Copper
               albedo                       // Other
            );

    return f0[min(alpha2BlockID(texCol), 3)].rgb * getLuma(texCol.rgb);
}

vec3 getTextureNormal(vec2 uv, float resolution, float scale) {
	vec2 texStep = 1.0 / resolution * vec2(2.0, 1.0);
    float height = length(texture2D(s_MatTexture, uv).rgb);
    vec2 dxy = height - vec2(length(texture2D(s_MatTexture, uv + vec2(texStep.x, 0.0)).rgb),
        length(texture2D(s_MatTexture, uv + vec2(0.0, texStep.y)).rgb));
    
	return normalize(vec3(dxy * scale / texStep, 1.0));
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

float getStars(const vec3 pos) {
    vec3 p = floor((normalize(pos) + 16.0) * 265.0);
    float stars = smoothstep(0.998, 1.0, hash13(p));

    return stars;
}

float drawSun(const vec3 pos) {
	return inversesqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
}

float getAO(vec4 vertexCol, const float shrinkLevel) {
    float lum = vertexCol.g * 2.0 - (vertexCol.r < vertexCol.b ? vertexCol.r : vertexCol.b);

    return min(lum + (1.0 - shrinkLevel), 1.0);
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

#define SKYLIGHT_INTENSITY 2.0
#define SUNLIGHT_INTENSITY 3.0
#define SUNSETLIGHT_INTENSITY 18.0
#define MOONLIGHT_INTENSITY 2.0
#define TORCHLIGHT_INTENSITY 7.5

#define WATER_COL vec3(0.0, 0.015, 0.02)
#define SKYLIT_COL vec3(0.9, 0.98, 1.0)
#define SUNLIT_COL vec3(1.0, 0.92, 0.8)
#define TORCHLIT_COL vec3(1.0, 0.65, 0.3)
#define MOONLIT_COL vec3(0.75, 0.8, 1.0)

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
		albedo.rgb = mix(albedo.rgb, getF0(texCol, albedo).rgb, 1.0);
	}
#endif

float time = getTimeFromFog(v_fog);
float rain = smoothstep(0.5, 0.3, fogColtrol.x);
vec3 sunMoonPos = (time > 0.0 ? 1.0 : -1.0) * SUN_MOON_DIR * vec3(cos(time), sin(time), -cos(time));
float outdoor = smoothstep(0.92, 0.94, v_lightmapUV.y);
vec3 worldNormal = normalize(cross(dFdx(chunkPos), dFdy(chunkPos)));

float reflectance = 0.0;
if (waterFlag > 0.5) {
    worldNormal = mul(getWaterWavesNormal(chunkPos.xz, frameTime), getTBNMatrix(worldNormal));
	reflectance = 1.0;
} else if (isMetallic) {
    worldNormal = mul(getTextureNormal(v_texcoord0, 2048.0, 0.0008), getTBNMatrix(worldNormal));
    reflectance = 0.5;
}

vec3 refPos = reflect(normalize(relPos), worldNormal);

float daylight = max(0.0, time);
float diffuse = max(0.0, dot(sunMoonPos, worldNormal));
float dirLight = mix(0.0, diffuse, outdoor);
float torchLit = v_lightmapUV.x * v_lightmapUV.x * v_lightmapUV.x * v_lightmapUV.x;
torchLit = mix(0.0, torchLit, smoothstep(0.96, 0.7, v_lightmapUV.y * daylight));

float skyBrightness = mix(1.0, 2.0, smoothstep(0.0, 0.1, daylight));
vec3 totalSky = contrastFilter(toneMapReinhard(getAtmosphere(normalize(relPos), sunMoonPos, vec3(0.4, 0.65, 1.0), skyBrightness)), 1.8);
// vec3 ambientLightCol = v_fog.rgb + (1.0 - max(max(v_fog.r, v_fog.g), v_fog.b));
vec3 ambientLightCol = mix(mix(mix(vec3(0.3, 0.3, 0.3), mix(vec3(0.8, 0.8, 0.8), vec3(1.0, 1.0, 1.0), outdoor), v_lightmapUV.y), TORCHLIT_COL, torchLit), mix(MOONLIT_COL, mix(SKYLIT_COL, SUNLIT_COL, 0.5), daylight), outdoor);
// ambientLightCol += 1.0 - max(max(ambientLightCol.r, ambientLightCol.g), ambientLightCol.b);
#ifndef SEASONS
    ambientLightCol = mix(vec3(0.3, 0.3, 0.3), ambientLightCol, getAO(v_color0, 0.65));
#endif
vec3 defaultCol = vec3(1.0, 1.0, 1.0);

vec3 lit = vec3(1.0, 1.0, 1.0);

lit *= mix(defaultCol, ambientLightCol, 1.0);
lit *= mix(defaultCol, SKYLIGHT_INTENSITY * SKYLIT_COL, dirLight * daylight * max(0.5, 1.0 - rain));
lit *= mix(defaultCol, SUNLIGHT_INTENSITY * SUNLIT_COL, dirLight * daylight * max(0.5, 1.0 - rain));
lit *= mix(defaultCol, MOONLIGHT_INTENSITY * MOONLIT_COL, dirLight * (1.0 - daylight) * max(0.5, 1.0 - rain));
lit *= mix(defaultCol, TORCHLIGHT_INTENSITY * TORCHLIT_COL, torchLit);

albedo.rgb *= lit;
albedo.rgb = uncharted2ToneMap(albedo.rgb, 2.4);
albedo.rgb = contrastFilter(albedo.rgb, 1.45);

if (waterFlag > 0.5 || isMetallic) {
    float cosTheta = clamp(1.0 - abs(dot(normalize(relPos), worldNormal)), 0.0, 1.0);

    if (waterFlag > 0.5) {
        albedo.a = mix(0.0, 1.0, cosTheta);
        albedo.rgb = WATER_COL;
    }

    if (bool(outdoor)) {
        vec3 reflectedSky = contrastFilter(toneMapReinhard(getAtmosphere(refPos, sunMoonPos, vec3(0.4, 0.65, 1.0), skyBrightness)), 1.8);
        albedo.rgb = mix(albedo.rgb, reflectedSky, outdoor * reflectance);

        float drawSpace = max(0.0, length(refPos.xz / (refPos.y * float(16))));
        if (drawSpace < 1.0 && !bool(step(refPos.y, 0.0))) {
            float clouds = renderClouds(refPos, sunMoonPos, rain, frameTime).x;
            float shade = renderClouds(refPos, sunMoonPos, rain, frameTime).y;

            float cloudBrightness = clamp(dot(reflectedSky, vec3(0.2126, 0.7152, 0.0722)) * 2.0, 0.0, 1.0);
            vec3 cloudCol = vec3(cloudBrightness, cloudBrightness, cloudBrightness);
            cloudCol *= mix(1.0, 0.5, shade);

            albedo.rgb = mix(albedo.rgb, mix(albedo.rgb, cloudCol, mix(clouds * 0.5, 0.0, drawSpace)), outdoor * reflectance);
            albedo += getStars(refPos) * (1.0 - daylight) * 0.5;
            albedo += drawSun(cross(refPos, sunMoonPos) * 25.0) * outdoor * (1.0 - drawSpace);
        }
    }
}

// vec3 fogCol = ambientLightCol + (1.0 - max(max(ambientLightCol.r, ambientLightCol.g), ambientLightCol.b));
vec3 fogCol = mix(v_fog.rgb, totalSky, v_lightmapUV.y);
albedo.rgb = mix(albedo.rgb, mix(fogCol, vec3(getLuma(fogCol), getLuma(fogCol), getLuma(fogCol)), rain), v_fog.a);

    gl_FragColor = albedo;
}
