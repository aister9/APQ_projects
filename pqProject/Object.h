#pragma once
#include "glHeaders.h"

namespace AISTER_GRAPHICS_ENGINE {
	class Object_t {
	public:
		glm::vec3 position;
		glm::quat rotation;
		glm::vec3 scale;

		Object_t() {
			position = glm::vec3(0, 0, 0);
			rotation = glm::quat(glm::vec3(0, 0, 0));
			scale = glm::vec3(1, 1, 1);
		}

		glm::vec3 getEulerRotation() {
			return glm::eulerAngles(rotation);
		}

		glm::mat4 getTRS() {
			return glm::translate(glm::mat4(1.0f), position) * glm::toMat4(rotation) * glm::scale(glm::mat4(1.0f), scale);
		}
	};
}