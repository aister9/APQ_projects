#version 460 core
layout(location = 0) in vec3 vertexPosition_modelspace;

uniform mat4 MVP;

void main() {
	vec4 p = vec4(vertexPosition_modelspace, 1);
	gl_Position = MVP * p;
}
