#version 460 core

uniform vec4 mtlColor;

uniform bool drawDepth;

uniform sampler2D textures;

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
	
	if(!drawDepth)
		color = texture(textures, TexCoord);
	else {
		// depth = LinearizeDepth(gl_FragCoord.z) / far; // divide by far for demonstration
		float depth = gl_FragCoord.w; // divide by far for demonstration
		color = vec4(vec3(depth), 1.0);
	}
}
