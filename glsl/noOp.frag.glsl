#version 330 core

uniform sampler2D u_Texture;
uniform vec2 u_ScreenDims;
uniform int u_Iterations;

in vec3 fs_Pos;
in vec2 fs_UV;

out vec4 out_Col;

vec4 reinhardOp(vec4 c)
{
    return c / (1.f + c);
}

vec4 gammaCorrection(vec4 c)
{
    return pow(c, vec4(1.f / 2.2f));
}

void main()
{
    vec4 color = texture(u_Texture, fs_UV);

    color = reinhardOp(color);
    color = gammaCorrection(color);

    out_Col = color;
}
