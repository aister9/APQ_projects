#version 460 core

in vec3 vertexPos;
uniform vec4 mtlColor;
uniform vec4 _edgeColor;
uniform float _thickNess;

uniform vec3 _center;
uniform vec3 _size;

out vec4 color;


void main() {
	vec3 min = _center - _size / 2;
	vec3 max = _center + _size / 2;

	if (vertexPos.x > min.x + _thickNess && vertexPos.x < max.x - _thickNess) {

		if (vertexPos.y > min.y + _thickNess && vertexPos.y < max.y - _thickNess) {

			color = mtlColor;

		}
		else if (vertexPos.z > min.z + _thickNess && vertexPos.z < max.z - _thickNess) {

			color = mtlColor;

		}
		else {

			color = _edgeColor;
		}
	}
	else {
		if (vertexPos.y > min.y + _thickNess && vertexPos.y < max.y - _thickNess) {
			if (vertexPos.z > min.z + _thickNess && vertexPos.z < max.z - _thickNess) {

				color = mtlColor;

			}
			else {

				color = _edgeColor;
			}
		}
		else {

			color = _edgeColor;
		}
	}

}
