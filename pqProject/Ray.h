#pragma once
#include "glHeaders.h"
#include "QBVH4.h"
#include <limits>
#include <stack>

namespace AISTER_GRAPHICS_ENGINE {
	class RayHit {
	public:
		glm::vec3 position;
		float distance;
		glm::vec3 normal;
		bool isHit;

		RayHit() {
			position = glm::vec3(0, 0, 0);
			distance = std::numeric_limits<float>::max();
			normal = glm::vec3(0, 0, 0);
			isHit = false;
		}

		RayHit(glm::vec3 pos, float dist) : position(pos), distance(dist), isHit(true) {
			normal = glm::vec3(0, 0, 0);
		}
	};

	class Ray {
	public:
		glm::vec3 origin;
		glm::vec3 direction;

		Ray() {
			origin = glm::vec3(0, 0, 0);
			direction = glm::vec3(0, 0, 0);
		}

		Ray(glm::vec3 origin, glm::vec3 dir) : origin(origin), direction(dir) {		}

		RayHit intersect(QBVH4Node node, int boxIdx) {
			glm::vec3 bMin = glm::vec3(node.start[0] + node.extent[0] * node.boxData[boxIdx].qlower[0],
										node.start[1] + node.extent[1] * node.boxData[boxIdx].qlower[1],
										node.start[2] + node.extent[2] * node.boxData[boxIdx].qlower[2]);
			glm::vec3 bMax = glm::vec3(node.start[0] + node.extent[0] * node.boxData[boxIdx].qupper[0],
										node.start[1] + node.extent[1] * node.boxData[boxIdx].qupper[1],
										node.start[2] + node.extent[2] * node.boxData[boxIdx].qupper[2]);

			glm::vec3 invDir = glm::vec3(1 / direction.x, 1 / direction.y, 1 / direction.z);

			glm::vec3 tMin = (bMin - origin) * invDir;
			glm::vec3 tMax = (bMax - origin) * invDir;

			glm::vec3 tmp1 = glm::min(tMin, tMax);
			glm::vec3 tmp2 = glm::max(tMin, tMax);

			tMin = tmp1; tMax = tmp2;

			float _min = glm::max(tMin.x, tMin.y);
			float _max = glm::min(tMax.x, tMax.y);

			if (tMin.x > tMax.y || tMin.y > tMax.x) { return RayHit(); }
			if (_min > tMax.z || tMin.z > _max) return RayHit();

			_min = glm::max(_min, tMin.z);
			_max = glm::min(_max, tMax.z);

			glm::vec3 minPts = origin + _min * direction;
			glm::vec3 maxPts = origin + _max * direction;

			if (glm::distance(origin, minPts) > glm::distance(origin, maxPts))
				std::swap(_min, _max);

			return RayHit(origin + _min * direction, _min);
		}

		RayHit intersect(std::vector<glm::vec3> &pts, __Tri_t &tri) {
			glm::vec3 e1 = pts[tri.ind[1]] - pts[tri.ind[0]];
			glm::vec3 e2 = pts[tri.ind[2]] - pts[tri.ind[0]];

			glm::vec3 h = glm::cross(direction, e2);
			float a = glm::dot(e1, h);

			float epsillon = 0.0000001;

			if (a > -epsillon && a < epsillon) return RayHit();

			float f = 1.0f / a;
			glm::vec3 s = origin - pts[tri.ind[0]];
			float u = f * glm::dot(s, h);

			if (u < 0.0 || u>1.0) return RayHit();

			glm::vec3 q = glm::cross(s, e1);
			float v = f * glm::dot(direction, q);

			if (v < 0.0 || u + v > 1.0) return RayHit();

			float t = f * glm::dot(e2, q);

			if (t > epsillon) {
				return RayHit(origin + direction * t, t);
			}
			else
				return RayHit();
		}

		RayHit traverse(std::vector<QBVH4Node> bvh, std::vector<glm::vec3> &pts, std::vector<__Tri_t> &triList) {
			std::stack<int> indStack;
			
			indStack.push(0);

			RayHit res;
			while (!indStack.empty()) {
				int dataInd = indStack.top();
				indStack.pop();

				for (int i = 0; i < 4; i++) {
					if ((bvh[dataInd].childFlag & (1 << i)) == (1 << i)) // is null
					{
						continue; // then skip
					}

					if (((bvh[dataInd].childFlag >> 4) & (1 << i)) == (1 << i)) // if leaf
					{
						RayHit check = intersect(pts, triList[bvh[dataInd].childs[i].triIdx]);
						if (check.isHit && check.distance < res.distance)
						{
							res = check;
						}
					}
					else {
						RayHit check = intersect(bvh[dataInd], i);
						if (check.isHit && check.distance < res.distance)
							indStack.push(dataInd+bvh[dataInd].childs[i].boxIdx);
					}
				}
			}

			return res;
		}
	};
}

std::vector<AISTER_GRAPHICS_ENGINE::RayHit> RayTraverse(
	glm::vec3											origin,
	std::vector<glm::vec3>								dirlist,
	std::vector<AISTER_GRAPHICS_ENGINE::QBVH4Node>		bvh,
	std::vector<glm::vec3>								vertices,
	std::vector<AISTER_GRAPHICS_ENGINE::__Tri_t>		trilist,
	int													vsize,
	int													tsize,
	int													width,
	int													height);