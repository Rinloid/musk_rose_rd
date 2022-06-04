$input inPos, inUV

$output uv, pos

#include "bgfx_shader.sh"

void main() {
uv = inUV;
pos = inPos.xz;

    gl_Position = mul(u_modelViewProj, vec4(inPos, 1.0));
}