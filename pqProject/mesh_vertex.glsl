#version 460 core
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec2 uv_coordinates;

uniform mat4 MVP;

out vec2 TexCoord;

void main() {
	vec4 p = vec4(vertexPosition_modelspace, 1);
	gl_Position = MVP * p;
	TexCoord = uv_coordinates;
}
