#version 460 core

uniform vec4 maskColor;

out vec4 color;
in vec2 TexCoord;

float near = 0.1;
float far = 100.0;

float LinearizeDepth(float depth)
{
	float z = depth * 2.0 - 1.0; // back to NDC 
	return (2.0 * near * far) / (far + near - z * (far - near));
}

void main() {

	color = maskColor;
}
