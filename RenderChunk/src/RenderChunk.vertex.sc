$input inCol, inPos, inUV0, inUV1
#ifdef INSTANCING
    $input i_data0, i_data1, i_data2, i_data3
#endif
$output col, fog, uv0, uv1, waterFlag, relPos, chunkPos, frameTime

#include <bgfx_shader.sh>

uniform vec4 RenderChunkFogAlpha;
uniform vec4 FogAndDistanceControl;
uniform vec4 ViewPositionAndTime;
uniform highp vec4 FogColor;

bool isPlant(vec4 vertexCol, vec4 pos) {
    vec3 fractPos = fract(pos.xyz);
    #if defined(ALPHA_TEST)
        return (vertexCol.g != vertexCol.b && vertexCol.r < vertexCol.g + vertexCol.b) || (fractPos.y == 0.9375 && (fractPos.z == 0.0 || fractPos.x == 0.0));
    #else
        return false;
    #endif
}

void main() {
waterFlag = 0.0;

mat4 model;
#ifdef INSTANCING
    model = mtxFromCols(i_data0, i_data1, i_data2, i_data3);
#else
    model = u_model[0];
#endif

vec3 worldPos = mul(model, vec4(inPos, 1.0)).xyz;
vec4 color;

#ifdef RENDER_AS_BILLBOARDS
    worldPos += vec3(0.5, 0.5, 0.5);
    vec3 viewDir = normalize(worldPos - ViewPositionAndTime.xyz);
    vec3 boardPlane = normalize(vec3(viewDir.z, 0.0, -viewDir.x));
    worldPos = (worldPos -
        ((((viewDir.yzx * boardPlane.zxy) - (viewDir.zxy * boardPlane.yzx)) *
        (inCol.z - 0.5)) +
        (boardPlane * (inCol.x - 0.5))));
    color = vec4(1.0, 1.0, 1.0, 1.0);
#else
    color = inCol;
#endif

vec3 modelCamPos = (ViewPositionAndTime.xyz - worldPos);
float camDis = sqrt(dot(modelCamPos, modelCamPos));
vec4 fogColor;
fogColor.rgb = FogColor.rgb;
fogColor.a = clamp(((((camDis / FogAndDistanceControl.z) + RenderChunkFogAlpha.x) -
    FogAndDistanceControl.x) / (FogAndDistanceControl.y - FogAndDistanceControl.x)), 0.0, 1.0);

#ifdef TRANSPARENT
    if (inCol.a < 0.95) {
        color.a = mix(inCol.a, 1.0, clamp((camDis / FogAndDistanceControl.w), 0.0, 1.0));
        waterFlag = 1.0;
    } else {
        waterFlag = 0.0;
    }
#endif

if (isPlant(inCol, vec4(inPos, 1.0))) {
    vec3 wavPos = abs(inPos.xyz - 8.0);
    float wave = sin(ViewPositionAndTime.w * 3.5 + 2.0 * wavPos.x + 2.0 * wavPos.z + wavPos.y);

    worldPos.x += wave * 0.03 * smoothstep(0.7, 1.0, inUV1.y);
}

uv0 = inUV0;
uv1 = inUV1;
col = color;
fog = fogColor;
relPos = worldPos;
chunkPos = inPos;
frameTime = ViewPositionAndTime.w;

    gl_Position = mul(u_viewProj, vec4(worldPos, 1.0));
}
