#pragma once
#include "glHeaders.h"
#include <vector>

namespace AISTER_GRAPHICS_ENGINE {

	typedef struct __Tri_t { int ind[3]; };

	typedef struct Box3ui8 { // 6byte
		UINT8 qlower[3];
		UINT8 qupper[3];
	};

	class QBVH4Node { // 12+12+24+1+16 byte ==> than 68 byte (because 4byte)
	public:
		glm::vec3 start; // 12 byte
		glm::vec3 extent; // 12 byte
		Box3ui8 boxData[4]; // 24 byte
		UINT8 childFlag; // 0000 0000 : isLeaf / isNull // 1byte

		union child { // 4byte
			UINT32 boxIdx; //offset next node index
			UINT32 triIdx; //offset triangle index
		};
		child childs[4]; // 16 byte

		QBVH4Node() { //null
			childFlag = 15;
		}

		size_t getSize_t() {
			return sizeof(QBVH4Node);
		}

		glm::vec3 getCenter() {
			return (start + (start + extent * 255.f)) * 0.5f;
		}

		glm::vec3 getSize() {
			return extent * 255.f;
		}

		int getLongestAxis() {
			glm::vec3 boxsize = extent * 255.f;

			if (boxsize.x > boxsize.y) {
				if (boxsize.x > boxsize.z) return 0; // x>y>z
				else return 2; // z>x>y
			}
			else
				if (boxsize.y > boxsize.z) return 1; //y> (z,x)
				else return 2; // z>y>x
		}

		bool operator<(QBVH4Node& target) {
			return start[getLongestAxis()] < target.start[target.getLongestAxis()];
		}

	};
}

std::vector<AISTER_GRAPHICS_ENGINE::__Tri_t> build(std::vector<glm::vec3>& pts, std::vector<int> triList, glm::mat4 trs);
std::vector<AISTER_GRAPHICS_ENGINE::QBVH4Node> build(std::vector<glm::vec3>& pts, std::vector<AISTER_GRAPHICS_ENGINE::__Tri_t> triList, glm::mat4 trs);
