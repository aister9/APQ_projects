#pragma once
#include "Object.h"
#include "glHeaders.h"

namespace AISTER_GRAPHICS_ENGINE {
	class Camera : public Object_t {
	public:
		glm::vec3 direction = glm::vec3(-1,0,0);
		glm::vec3 up = glm::vec3(0, 1, 0);
		float fovy = 38.f;

		glm::vec2 screenResolution;

		float _near = 0.1f;
		float _far = 4.5f;

		glm::mat4 getProjectionMatrix() {

			return glm::perspective(glm::radians(fovy), (float)screenResolution.x / (float)screenResolution.y, _near, _far) * glm::mat4(1.0f);
		}

		glm::mat4 getViewMatrix() {
			//direction = glm::vec3(getTRS() * glm::vec4(direction,1));
			return glm::lookAt(position, position+direction, up) * glm::mat4(1.0f);
		}
		glm::mat4 getViewMatrix(glm::vec3 target, glm::vec3 up) {

			this->direction = glm::normalize(position - target);
			this->up = up;

			return glm::lookAt(position, target, up) * glm::mat4(1.0f);
		}

		void setFOV_K(float fx, float fy, int w, int h, float znear, float zfar) {
			float fovy = 2 * glm::atan(h / (2 * fy));
			screenResolution.x = w;
			screenResolution.y = h;
			_near = znear;
			_far = zfar;
		}

		glm::vec3 getHorizontal() {
			const float aspect = screenResolution.x / screenResolution.y;
			const float cosFovy = 0.66f;
			return cosFovy * aspect * glm::normalize(glm::cross(direction, up));
		}

		glm::vec3 getVertical() {
			const float cosFovy = 0.66f;
			return cosFovy * glm::normalize(glm::cross(getHorizontal(), direction));
		}

	};

	glm::mat4 getPerspectiveUsingK(float fx, float fy, float cx, float cy, int w, int h, float znear, float zfar) {
		float fovy = 2 * glm::atan(h / (2 * fy));

		return glm::perspective(fovy, (float)w / (float)h, znear, zfar);
	}
}