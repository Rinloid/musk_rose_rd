$input fog, relPos, frameTime, skyCol, fogCol

#include <bgfx_shader.sh>
#include <functions.sh>

#define SKY_COL vec4(0.34, 0.51, 0.64, 1.0)

void main() {
vec4 albedo = fogCol * SKY_COL;

float cloudBrightness = clamp(dot(albedo.rgb, vec3(0.2126, 0.7152, 0.0722)) * 2.0, 0.0, 1.0);
vec3 cloudCol = vec3(cloudBrightness, cloudBrightness, cloudBrightness);

vec2 cloudPos = relPos.xz * 10.0;

float clouds = getBlockyClouds(cloudPos * 2.0, 0.0, frameTime).x;
float shade = getBlockyClouds(cloudPos * 2.0, 0.0, frameTime).y;

cloudCol *= mix(1.02, 0.75, shade);

albedo.rgb = mix(albedo.rgb, cloudCol, clouds);
albedo = mix(albedo, fogCol, fog);

    gl_FragColor = albedo;
}