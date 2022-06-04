$input col, fog, uv0, uv1, waterFlag, relPos, chunkPos, frameTime

#include <bgfx_shader.sh>
#include <functions.sh>

uniform vec4 FogColor;
SAMPLER2D(s_MatTexture, 0);
SAMPLER2D(s_LightMapTexture, 1);
SAMPLER2D(s_SeasonsTexture, 2);

#define SUN_MOON_DIR vec3(-0.5, 1.0, 0.0)
#define SKYLIGHT_INTENSITY 2.0
#define SUNLIGHT_INTENSITY 5.0
#define SUNSETLIGHT_INTENSITY 18.0
#define MOONLIGHT_INTENSITY 2.2
#define TORCHLIGHT_INTENSITY 7.5
#define CLOUD_RENDER_DISTAMCE 16
#define SKY_COL vec4(0.34, 0.51, 0.64, 1.0)

void main() {
const vec3 skyLitCol = vec3(1.0, 1.0, 1.0);
const vec3 sunLitCol = vec3(1.0, 0.92, 0.75);
const vec3 sunSetLitCol = vec3(1.0, 0.85, 0.3);
const vec3 torchLitCol = vec3(1.0, 0.65, 0.3);
const vec3 moonLitCol = vec3(0.56, 0.60, 0.98);
const vec3 shadowCol = vec3(1.05, 1.08, 1.2);
const vec3 waterCol = vec3(0.0, 0.3, 0.6);

vec4 albedo;
vec3 worldNormal = normalize(cross(dFdx(chunkPos), dFdy(chunkPos)));
if (waterFlag > 0.5) {
    worldNormal = mul(getWaterWavesNormal(chunkPos.xz, frameTime), getTBNMatrix(worldNormal));
}

#if defined(DEPTH_ONLY_OPAQUE) || defined(DEPTH_ONLY)
    albedo.rgb = vec3(1.0, 1.0, 1.0);
#else
    albedo = texture2D(s_MatTexture, uv0);

    #if defined(ALPHA_TEST) || defined(DEPTH_ONLY)
        if (albedo.a < 0.5) {
            discard;
        }
    #endif

    #if defined(SEASONS) && (defined(OPAQUE) || defined(ALPHA_TEST))
        albedo.rgb *= mix(vec3(1.0, 1.0, 1.0), texture2D(s_SeasonsTexture, col.xy).rgb * 2.0, col.b);
        albedo.rgb *= col.aaa;
    #else
        if (abs(col.r - col.g) > 0.001 || abs(col.g - col.b) > 0.001) {
            albedo.rgb *= normalize(col.rgb);
        }

        albedo.rgb *= getAO(col, 0.4);
    #endif /* defined(SEASONS) && (defined(OPAQUE) || defined(ALPHA_TEST)) */
#endif /* defined(DEPTH_ONLY_OPAQUE) || defined(DEPTH_ONLY) */

#ifndef TRANSPARENT
    albedo.a = 1.0;
#endif

albedo.rgb *= texture2D(s_LightMapTexture, uv1).rgb;

vec3 reflectPos = reflect(normalize(relPos), worldNormal);
float daylight = smoothstep(0.4, 1.0, texture2D(s_LightMapTexture, vec2(0.0, 1.0)).r);
float time = getTimeFromLightmap(daylight);
// float time = getTimeFromFog(FogColor);
// vec3 sunMoonPos = (time > 0.0 ? 1.0 : -1.0) * SUN_MOON_DIR * vec3(cos(time), sin(time), -cos(time));
vec3 sunMoonPos = SUN_MOON_DIR;
sunMoonPos.y = daylight > 0.0 ? smoothstep(0.5, 1.0, daylight) : 1.0 - daylight;
vec3 fogCol = fog.rgb;
float outdoor = smoothstep(0.92, 0.96, uv1.y);
float diffuse = max(0.0, dot(sunMoonPos, worldNormal));
float dirLight = mix(0.0, diffuse, outdoor);
float torchLit = uv1.x * uv1.x * uv1.x * uv1.x;
torchLit = mix(0.0, torchLit, smoothstep(0.96, 0.7, uv1.y * daylight));

vec3 lit = vec3(1.0, 1.0, 1.0);

vec3 ambientLightCol = fog.rgb + (1.0 - max(max(fog.r, fog.g), fog.b));
lit *= mix(vec3(1.0, 1.0, 1.0), ambientLightCol, 1.0); // Ambient light
lit *= mix(vec3(1.0, 1.0, 1.0), SKYLIGHT_INTENSITY * skyLitCol, dirLight * daylight);
lit *= mix(vec3(1.0, 1.0, 1.0), SUNLIGHT_INTENSITY * sunLitCol, dirLight * daylight);
lit *= mix(vec3(1.0, 1.0, 1.0), MOONLIGHT_INTENSITY * moonLitCol, dirLight * (1.0 - daylight));
lit *= mix(vec3(1.0, 1.0, 1.0), TORCHLIGHT_INTENSITY * torchLitCol, torchLit);

albedo.rgb *= lit;

albedo.rgb = uncharted2ToneMap(albedo.rgb, 2.4);
albedo.rgb = contrastFilter(albedo.rgb, 1.45);

if (waterFlag > 0.5) {
    float cosTheta = clamp(1.0 - abs(dot(normalize(relPos), worldNormal)), 0.0, 1.0);

    albedo.a = mix(0.1, 0.9, cosTheta);

    vec3 skyCol = fogCol * SKY_COL.rgb;
    albedo.rgb = mix(mix(skyCol, fogCol, smoothstep(0.8, 1.0, 1.0 - reflectPos.y)), waterCol, outdoor);

    float drawSpace = max(0.0, length(reflectPos.xz / (reflectPos.y * float(CLOUD_RENDER_DISTAMCE))));
    if (drawSpace < 1.0 && !bool(step(reflectPos.y, 0.0)) && bool(outdoor)) {
        float cloudBrightness = clamp(dot(skyCol, vec3(0.2126, 0.7152, 0.0722)) * 2.0, 0.0, 1.0);
        vec3 cloudCol = vec3(cloudBrightness, cloudBrightness, cloudBrightness);

        vec2 cloudPos = reflectPos.xz / reflectPos.y;

        float clouds = getBlockyClouds(cloudPos * 2.0, 0.0, frameTime).x * outdoor;
        float shade = getBlockyClouds(cloudPos * 2.0, 0.0, frameTime).y;

        cloudCol *= mix(1.02, 0.75, shade);

        albedo.rgb = mix(albedo.rgb, mix(albedo.rgb, cloudCol, clouds), cosTheta);
    }

    albedo += drawSun(cross(reflectPos, sunMoonPos) * 20.0) * daylight * outdoor;
}

albedo.rgb = mix(albedo.rgb, fog.rgb, fog.a);

    gl_FragColor = albedo;

} /* main */
