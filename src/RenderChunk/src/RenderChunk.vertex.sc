$input a_color0, a_position, a_texcoord0, a_texcoord1
#ifdef INSTANCING
    $input i_data0, i_data1, i_data2, i_data3
#endif

$output v_color0, v_fog, v_texcoord0, v_lightmapUV, relPos, fragPos, frameTime, waterFlag, fogControl

#include <bgfx_compute.sh>

uniform vec4 RenderChunkFogAlpha;
uniform vec4 FogAndDistanceControl;
uniform vec4 ViewPositionAndTime;
uniform vec4 FogColor;

bool isPlant(const vec4 col, const vec3 pos) {
    vec3 fractPos = fract(pos.xyz);
#   if defined(ALPHA_TEST)
        return (col.g != col.b && col.r < col.g + col.b) || (fractPos.y == 0.9375 && (fractPos.z == 0.0 || fractPos.x == 0.0));
#   else
        return false;
#   endif
}

void main() {
mat4 model =
#ifdef INSTANCING
    mtxFromCols(i_data0, i_data1, i_data2, i_data3);
#else
    u_model[0];
#endif

relPos = mul(model, vec4(a_position, 1.0)).xyz;

v_color0 =
#ifdef RENDER_AS_BILLBOARDS
    vec4(1.0, 1.0, 1.0, 1.0);
    relPos += vec3(0.5, 0.5, 0.5);
    vec3 viewDir = normalize(relPos - ViewPositionAndTime.xyz);
    vec3 boardPlane = normalize(vec3(viewDir.z, 0.0, -viewDir.x));
    relPos = (relPos - ((((viewDir.yzx * boardPlane.zxy) - (viewDir.zxy * boardPlane.yzx)) * (a_color0.z - 0.5)) + (boardPlane * (a_color0.x - 0.5))));
#else
    a_color0;
#endif

if (isPlant(a_color0, a_position)) {
    vec3 wavPos = abs(a_position.xyz - 8.0);
    float wave = sin(ViewPositionAndTime.w * 3.5 + 2.0 * wavPos.x + 2.0 * wavPos.z + wavPos.y);

    relPos.x += wave * 0.05 * smoothstep(0.7, 1.0, a_texcoord1.y);
}

fogControl = FogAndDistanceControl.xy;

v_fog.rgb = FogColor.rgb;

float fogStart = 0.0;
float fogEnd = 1.2;

v_fog.a = clamp(((((length((ViewPositionAndTime.xyz - relPos)) / FogAndDistanceControl.z) + RenderChunkFogAlpha.x) - fogStart) / (fogEnd - fogStart)), 0.0, 1.0);

v_texcoord0 = a_texcoord0;
v_lightmapUV = a_texcoord1;

fragPos = a_position;
frameTime = ViewPositionAndTime.w;

waterFlag = 0.0;
#ifdef TRANSPARENT
    if (a_color0.r != a_color0.g || a_color0.g != a_color0.b || a_color0.r != a_color0.b) waterFlag = 1.0;
#endif

    gl_Position = mul(u_viewProj, vec4(relPos, 1.0));
}
