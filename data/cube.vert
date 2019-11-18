

//Included by some_shader.glsl

layout(location = 0) in vec2 a_TexCoord;

// layout(location = 0) in mat4 a_Coord;
layout(location = 1) in vec4 a_VertColor;
layout(location = 2) in vec2 a_vertex_pos[4];


layout(location = 0) out VertexData {
    vec2 v_TexCoord;
    vec4 v_VertColor;

} outData;


layout(set = 0, binding = 0) uniform Locals {
    mat4 u_Transform;
};

// Wrong does not match the cooridiantes on client side. 
const vec2 square[4] = vec2[4](
    vec2(0.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0)
);


//Not available for spirv/vulkan
// uniform vec4 uv_lut[4] = vec4[4](
//     vec4(1.0, 0.0, 1.0, 0.0), 
//     vec4(1.0, 0.0, 0.0, 1.0), 
//     vec4(0.0, 1.0, 1.0, 0.0), 
//     vec4(0.0, 1.0, 0.0, 1.0)); 
// uniform Sample {
//     vec4 uv_lut[4] = vec4[4](
//     vec4(1.0, 0.0, 1.0, 0.0), // left bottom
//     vec4(1.0, 0.0, 0.0, 1.0), // left top
//     vec4(0.0, 1.0, 1.0, 0.0), // right bottom
//     vec4(0.0, 1.0, 0.0, 1.0)); // right top
// }

// const float PI = 3.1415926535897932384626433832795;
// const float PI_2 = 1.57079632679489661923;
// const float PI_4 = 0.785398163397448309616;

// vec2 sqrtf(float a){
//     if(a>=0.0) return vec2(sqrt(a), 0.0);
//     else return vec2(0.0, sqrt(-a));
// }
// float PHI = (1.0 + sqrtf(5.0).xy ) / 2.0;
// float PI180 = float(PI / 180.0);
// float sind(float a){return sin(a * PI180);}
// float cosd(float a){return cos(a * PI180);}

void main() {
    
    vec4 MVP = u_Transform * vec4(a_vertex_pos[gl_VertexIndex].xy, 0.0, 1.0);

    outData.v_VertColor = a_VertColor;
    outData.v_TexCoord = a_TexCoord;
    gl_Position = MVP;

}
