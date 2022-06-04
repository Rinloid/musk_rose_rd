$input uv, pos

#include "bgfx_shader.sh"

uniform highp vec4 SunMoonColor;
SAMPLER2D(s_SunMoonTexture, 0);

float drawSun(const vec2 pos) {
	return inversesqrt(pos.x * pos.x + pos.y * pos.y);
}

void main() {
vec4 albedo = texture2D(s_SunMoonTexture, uv);
if (texture2D(s_SunMoonTexture, vec2(0.5, 0.5)).r > 0.1) {
    albedo = mix(vec4(0.0, 0.0, 0.0, 0.0), vec4(1.0, 1.0, 1.0, 1.0), smoothstep(0.1, 1.0, drawSun(pos * 25.0)));
}
    gl_FragColor = albedo;
}