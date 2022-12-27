vec4 a_color0     : COLOR0;
vec3 a_position   : POSITION;
vec2 a_texcoord0  : TEXCOORD0;
vec2 a_texcoord1  : TEXCOORD1;

vec4 i_data0      : TEXCOORD8;
vec4 i_data1      : TEXCOORD7;
vec4 i_data2      : TEXCOORD6;
vec4 i_data3      : TEXCOORD5;

vec4 v_color0     : COLOR0;
vec4 v_fog        : COLOR2;
vec2 v_texcoord0  : TEXCOORD0;
vec2 v_lightmapUV : TEXCOORD1;
vec3 v_position   : TEXCOORD2;

vec3 relPos : RELATIVE_POSITION;
vec3 fragPos : FRAGMENT_POSITION;
float frameTime : FRAME_TIME_COUNTER;
float waterFlag : WATER_FLAG;
vec2 fogControl : FOG_CONTROL;