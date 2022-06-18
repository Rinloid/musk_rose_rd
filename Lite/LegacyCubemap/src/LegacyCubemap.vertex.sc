$input a_position

$output v_fog, relPos, frameTime, fogColtrol

#include <bgfx_shader.sh>

uniform vec4 FogColor;
uniform vec4 ViewPositionAndTime;
uniform vec4 FogAndDistanceControl;

void main() {
relPos = a_position;
relPos.y -= 0.2;
relPos.yz *= -1.0;
v_fog = FogColor;
frameTime = ViewPositionAndTime.w;
fogColtrol = FogAndDistanceControl.xy;

    gl_Position = mul(u_modelViewProj, vec4(a_position, 1.0));
}