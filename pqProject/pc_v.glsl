#version 460 core
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in float weights;

uniform mat4 MVP;

out float weight;

void main() {
	vec4 p = vec4(vertexPosition_modelspace, 1);
	gl_Position = MVP * p;
	weight = weights;
}
