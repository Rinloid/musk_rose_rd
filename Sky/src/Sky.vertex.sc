
$input inPos, inCol

$output fog, relPos, frameTime, skyCol, fogCol

#include <bgfx_shader.sh>

uniform highp vec4 SkyColor;
uniform highp vec4 FogColor;
uniform highp vec4 ViewPositionAndTime;

void main() {
highp vec4 pos = vec4(inPos, 1.0);
// pos.y -= length(pos.xz) * 0.2;

fog = inCol.r;
relPos = inPos;
frameTime = ViewPositionAndTime.w;
skyCol = SkyColor;
fogCol = FogColor;

    gl_Position = mul(u_modelViewProj, pos);
}