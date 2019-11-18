#version 450

layout(location = 0) in VertexDataz {
    vec2 v_TexCoord;
    vec4 v_VertColor;
} inData;
//Location for in must match vertex locations for out 
// layout(location = 0) in vec2 v_TexCoord;

layout(location = 0) out vec4 o_Target;


layout(set = 1, binding = 0) uniform texture2D u_texture; 

//Set defines which bind group.
//Bind defines which binding index is ued on a bind group in rust code
//Bind is unique per set(?) - confirm this before finalising parser. 
layout(set = 0, binding = 0) uniform sampler u_sampler;  

layout(set = 0, binding = 1) uniform Experiment {
    vec4 color;
    vec4 colorz;
};
void main() {
   vec4 tex = texture(sampler2D(u_texture, u_sampler), inData.v_TexCoord);
//    vec4 tint_color = vec4(1.0, 1.0, 1.0, 1.0); 
     vec4 tint_color = inData.v_VertColor; 
//    o_Target.rgb = tex.rgb * tint_color.rgb * tint_color.a * tex.a; 
//    o_Target.a  = tex.a * tint_color.a * 1.0; 


    o_Target = tint_color * tex; //Desaturate(tex.rgb, 0.5);
}
