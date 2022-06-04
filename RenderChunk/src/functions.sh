// https://www.shadertoy.com/view/4djSRW
float hash12(vec2 p) {
	vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

/*
 ** Simplex Noise modified by Rin
 ** Original author: Ashima Arts (MIT License)
 ** See: https://github.com/ashima/webgl-noise
 **      https://github.com/stegu/webgl-noise
*/

mat3 getTBNMatrix(const vec3 normal) {
    vec3 T = vec3(abs(normal.y) + normal.z, 0.0, normal.x);
    vec3 B = cross(T, normal);
    vec3 N = vec3(-normal.x, normal.y, normal.z);

    return mat3(T, B, N);
}

float mod289(float x) {
    return x - floor(x * 1.0 / 289.0) * 289.0;
}

vec2 mod289(vec2 x) {
    return x - floor(x * 1.0 / 289.0) * 289.0;
}

vec3 mod289(vec3 x) {
    return x - floor(x * 1.0 / 289.0) * 289.0;
}

vec4 mod289(vec4 x) {
    return x - floor(x * 1.0 / 289.0) * 289.0;
}

float permute289(float x) {
    return mod289((x * 34.0 + 1.0) * x);
}

vec3 permute289(vec3 x) {
    return mod289((x * 34.0 + 1.0) * x);
}

vec4 permute289(vec4 x) {
    return mod289((x * 34.0 + 1.0) * x);
}

float snoise(vec2 v) {
    const vec4 C = vec4(
        0.211324865405187,   // (3.0-sqrt(3.0))/6.0
        0.366025403784439,   // 0.5*(sqrt(3.0)-1.0)
        -0.577350269189626,  // -1.0 + 2.0 * C.x
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
    m = m*m;
    m = m*m;

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

float fBM(vec2 x, const float amp, const float lower, const float upper, const float time, const int octaves) {
    float v = 0.0;
    float amptitude = amp;

    for (int i = 0; i < octaves; i++) {
        v += amptitude * (snoise(x) * 0.5 + 0.5);

        if (v >= upper) {
            break;
        } else if (v + amptitude <= lower) {
            break;
        }

        x         *= 2.0;
        amptitude *= 0.5;
    }

	return smoothstep(lower, upper, v);
}

float getPlaneBlockyClouds(const vec2 pos, const float rain, const float time) {
    vec2 p = pos;
    p += time * 0.15;
    float body = hash12(floor(p));
    body = (body > mix(0.92, 0.0, rain)) ? 1.0 : 0.0;

    return body;
}

vec2 getBlockyClouds(const vec2 pos, const float rain, const float time) {
    const int steps = 48;
    const float stepSize = 0.008;

    float clouds = 0.0;
    float shade = 0.0;
    for (int i = 0; i < steps; i++) {
        float height = 1.0 + float(i) * stepSize;

        clouds += getPlaneBlockyClouds(pos * height, rain, time);
        shade = mix(shade, 1.0, clouds / float(steps) * float(steps) * stepSize);
    } clouds /= float(steps);

    //clouds = clamp(clouds * 10.0, 0.0, 1.0);
    clouds = clouds > 0.0 ? 1.0 : 0.0;

    return vec2(clouds, shade);
}

float getWaterWaves(vec2 p, const float time) {
	float r = 0.0;

    r += snoise(vec2(p.x - time * 0.4, p.y * 0.8 + time * 0.4) * 0.4) * 2.0; // Big waves
    r += snoise(vec2(p.x * 1.2 + time * 0.65, p.y - time * 0.75)) * 0.5; // Small waves

	return r * 0.005;
}

vec3 getWaterWavesNormal(const vec2 pos, const float time) {
	const float texStep = 0.04;
	float height = getWaterWaves(pos, time);
	vec2 dxy = height - vec2(getWaterWaves(pos + vec2(texStep, 0.0), time),
		getWaterWaves(pos + vec2(0.0, texStep), time));
    
	return normalize(vec3(dxy / texStep, 1.0));
}

float drawSun(const vec3 pos) {
	return inversesqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
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