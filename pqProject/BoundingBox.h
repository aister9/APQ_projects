#pragma once
#include "glHeaders.h"
#include "Object.h"
#include <vector>


namespace AISTER_GRAPHICS_ENGINE {
	class bbox : public Object_t {
	public:
		glm::vec3 center;
		glm::vec3 size;

		bbox() : center(glm::vec3(0, 0, 0)), size(glm::vec3(0, 0, 0)) {}
		bbox(glm::vec3 _center, glm::vec3 _size) : center(_center), size(_size) {}

		std::vector<glm::vec3> getCorner() {
			std::vector<glm::vec3> res = {
				glm::vec3(center.x - size.x / 2, center.y - size.y / 2, center.z - size.z / 2),
				glm::vec3(center.x + size.x / 2, center.y - size.y / 2, center.z - size.z / 2),
				glm::vec3(center.x - size.x / 2, center.y + size.y / 2, center.z - size.z / 2),
				glm::vec3(center.x + size.x / 2, center.y + size.y / 2, center.z - size.z / 2),
				glm::vec3(center.x - size.x / 2, center.y - size.y / 2, center.z + size.z / 2),
				glm::vec3(center.x + size.x / 2, center.y - size.y / 2, center.z + size.z / 2),
				glm::vec3(center.x - size.x / 2, center.y + size.y / 2, center.z + size.z / 2),
				glm::vec3(center.x + size.x / 2, center.y + size.y / 2, center.z + size.z / 2) };
			return res;
		};

		bool isNull() const  { return size == glm::vec3(0, 0, 0);
		}

		bbox operator+(bbox const& obj) {
			if (isNull() && obj.isNull()) return bbox();
			else if (isNull()) return obj;
			else if (obj.isNull()) return bbox(center, size);

			glm::vec3 bMin = glm::min(glm::min(center - size * 0.5f,
				obj.center - obj.size * 0.5f),
				glm::min(center + size * 0.5f,
					obj.center + obj.size * 0.5f));
			glm::vec3 bMax = glm::max(glm::max(center - size * 0.5f,
				obj.center - obj.size * 0.5f),
				glm::max(center + size * 0.5f,
					obj.center + obj.size * 0.5f));

			glm::vec3 _center = (bMin + bMax) * 0.5f;
			glm::vec3 _size = (bMax - bMin);

			return bbox(_center, _size);
		}
	};
}