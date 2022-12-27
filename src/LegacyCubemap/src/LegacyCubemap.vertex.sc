$input a_position

$output v_fog, relPos, frameTime, fogControl

#include <bgfx_compute.sh>

uniform vec4 FogColor;
uniform vec4 ViewPositionAndTime;
uniform vec4 FogAndDistanceControl;

void main() {
relPos = a_position;
relPos.y -= 0.2;
relPos.yz *= -1.0;
v_fog = FogColor;
frameTime = ViewPositionAndTime.w;
fogControl = FogAndDistanceControl.xy;

    gl_Position = mul(u_modelViewProj, vec4(a_position, 1.0));
}