#version 460 core

uniform vec4 mtlColor;

uniform float minweight;
uniform float maxweight;

in float weight;
out vec4 color;

vec4 getWeightColor(float w, float min, float max) {
	float nd = w / max;

	float red = 1, green = 1;
	if (nd <= 0.5)
	{
		red = 1, green = nd * 2;
	}
	else {
		green = 1; red = (1 - nd) * 2;
	}

	return vec4(red, green, 0, 0.2f);
}

void main() {
	color = getWeightColor(weight, minweight, maxweight);
}
