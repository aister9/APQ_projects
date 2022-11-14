#include "QBVH4.h"
#include <vector>
#include <algorithm>
#include <iostream>


int getLongestAxis(const glm::vec3 boxsize) {
	if (boxsize.x > boxsize.y) {
		if (boxsize.x > boxsize.z) return 0; // x>y>z
		else return 2; // z>x>y
	}
	else
		if (boxsize.y > boxsize.z) return 1; //y> (z,x)
		else return 2; // z>y>x
}

glm::vec3 getCenter(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) {
	glm::vec3 _min = glm::min(glm::min(p1, p2), p3);
	glm::vec3 _max = glm::max(glm::max(p1, p2), p3);
	
	return (_min + _max) * 0.5f;
}

glm::vec3 trspts(glm::mat4 trs, glm::vec3 pts) {
	glm::vec4 res = trs * glm::vec4(pts, 1);

	return glm::vec3(res);
}

std::vector<AISTER_GRAPHICS_ENGINE::__Tri_t> build(std::vector<glm::vec3>& pts, std::vector<int> triList, glm::mat4 trs) {
	//Preprocess make_trinagle_list
	std::vector<AISTER_GRAPHICS_ENGINE::__Tri_t> triangles;
	glm::vec4 bmin = trs*glm::vec4(pts[0],1), bmax = trs * glm::vec4(pts[0], 1);
	for (int i = 0; i < triList.size(); i += 3) {
		AISTER_GRAPHICS_ENGINE::__Tri_t t;
		t.ind[0] = triList[i + 0];
		t.ind[1] = triList[i + 1];
		t.ind[2] = triList[i + 2];

		bmin = glm::min(bmin, trs * glm::vec4(pts[t.ind[0]],1));
		bmin = glm::min(bmin, trs * glm::vec4(pts[t.ind[1]],1));
		bmin = glm::min(bmin, trs * glm::vec4(pts[t.ind[2]],1));
		 
		bmax = glm::max(bmax, trs * glm::vec4(pts[t.ind[0]],1));
		bmax = glm::max(bmax, trs * glm::vec4(pts[t.ind[1]],1));
		bmax = glm::max(bmax, trs * glm::vec4(pts[t.ind[2]],1));

		triangles.push_back(t);
	}

	glm::vec3 bsize = (glm::vec3(bmax) - glm::vec3(bmin));
	int longestAxis = getLongestAxis(bsize);


	//Preprocess trianlge list sorting
	std::sort(triangles.begin(), triangles.end(),
		[&longestAxis, &pts, &trs](AISTER_GRAPHICS_ENGINE::__Tri_t& a, AISTER_GRAPHICS_ENGINE::__Tri_t& b) -> bool
		{
			glm::vec3 aCenter = getCenter(trspts(trs ,pts[a.ind[0]]), trspts(trs, pts[a.ind[1]]), trspts(trs, pts[a.ind[2]]));
			glm::vec3 bCenter = getCenter(trspts(trs, pts[b.ind[0]]), trspts(trs, pts[b.ind[1]]), trspts(trs, pts[b.ind[2]]));

			return aCenter[longestAxis] > bCenter[longestAxis];
		}
	);

	return triangles;
}

std::vector<AISTER_GRAPHICS_ENGINE::QBVH4Node> build(std::vector<glm::vec3>& pts, std::vector<AISTER_GRAPHICS_ENGINE::__Tri_t> triList, glm::mat4 trs) {
	int sLength = triList.size();

	std::vector<AISTER_GRAPHICS_ENGINE::QBVH4Node> res;

	int treeDepth = 0;
	while (sLength > 1) {
		treeDepth++;
		const int pLength = sLength; // prev depth`s size
		sLength = ceil(sLength / 4.0f);

		std::cout << sLength << " ";

		std::vector<AISTER_GRAPHICS_ENGINE::QBVH4Node> curVec;

		if (res.size() == 0) // than leaf node
		{
			for (int i = 0; i < sLength; i++) { // make node
				AISTER_GRAPHICS_ENGINE::QBVH4Node q;
				
				glm::vec3 tLower[4], tUpper[4];
				glm::vec3 bmin = { 987654321,987654321, 987654321 }, bmax = { -987654321 ,-987654321 ,-987654321 };

				UINT8 leafMask = 15;
				UINT8 nullMask = 0;

				//set child
				for (int j = 0; j < 4; j++) {
					if (i * 4 + j >= pLength) { // out of memory
						nullMask |= (1 << j);
						continue;
					}
					q.childs[j].triIdx = i * 4 + j;

					tLower[j] = glm::min(glm::min(trspts(trs, pts[triList[i * 4 + j].ind[0]]), trspts(trs, pts[triList[i * 4 + j].ind[0]])), trspts(trs, pts[triList[i * 4 + j].ind[0]]));
					tUpper[j] = glm::max(glm::max(trspts(trs, pts[triList[i * 4 + j].ind[0]]), trspts(trs, pts[triList[i * 4 + j].ind[0]])), trspts(trs, pts[triList[i * 4 + j].ind[0]]));

					bmin = glm::min(tLower[j], bmin);
					bmax = glm::max(tLower[j], bmax);
				}

				glm::vec3 bsize = (bmax - bmin);
				glm::vec3 extent = bsize / 255.f;

				q.start = bmin;
				q.extent = extent;

				q.childFlag = (leafMask << 4) + nullMask;

				//set up box data
				for (int j = 0; j < 4; j++) {
					if (i * 4 + j >= pLength) { // out of memory
						break;
					}

					AISTER_GRAPHICS_ENGINE::Box3ui8 boxdata;
					boxdata.qlower[0] = (tLower[j].x - bmin.x) / extent.x;
					boxdata.qlower[1] = (tLower[j].y - bmin.y) / extent.y;
					boxdata.qlower[2] = (tLower[j].z - bmin.z) / extent.z;

					boxdata.qupper[0] = (tUpper[j].x - bmin.x) / extent.x;
					boxdata.qupper[1] = (tUpper[j].y - bmin.y) / extent.y;
					boxdata.qupper[2] = (tUpper[j].z - bmin.z) / extent.z;

					q.boxData[j] = boxdata;
				}

				curVec.push_back(q);
			}
			std::sort(curVec.begin(), curVec.end());

			res.insert(res.begin(), curVec.begin(), curVec.end());
		}
		else {
			for (int i = 0; i < sLength; i++) {
				AISTER_GRAPHICS_ENGINE::QBVH4Node q;

				UINT8 leafMask = 0;
				UINT8 nullMask = 0;

				glm::vec3 tLower[4], tUpper[4];
				glm::vec3 bmin = { 987654321,987654321, 987654321 }, bmax = { -987654321 ,-987654321 ,-987654321 };

				for (int j = 0; j < 4; j++) {
					if (i * 4 + j >= pLength) { // out of memory
						nullMask |= (1 << j);
						continue;
					}
					q.childs[j].boxIdx = i * 4 + j;

					tLower[j] = res[i*4+j].start;
					tUpper[j] = res[i*4+j].start + res[i*4+j].extent * 255.f;

					bmin = glm::min(tLower[j], bmin);
					bmax = glm::max(tUpper[j], bmax);
				}

				glm::vec3 bsize = (bmax - bmin);
				glm::vec3 extent = bsize / 255.f;

				q.start = bmin;
				q.extent = extent;

				q.childFlag = (leafMask << 4) + nullMask;

				//set up box data
				for (int j = 0; j < 4; j++) {
					if (i * 4 + j >= pLength) { // out of memory
						break;
					}

					AISTER_GRAPHICS_ENGINE::Box3ui8 boxdata;
					boxdata.qlower[0] = (tLower[j].x - bmin.x) / extent.x;
					boxdata.qlower[1] = (tLower[j].y - bmin.y) / extent.y;
					boxdata.qlower[2] = (tLower[j].z - bmin.z) / extent.z;

					boxdata.qupper[0] = (tUpper[j].x - bmin.x) / extent.x;
					boxdata.qupper[1] = (tUpper[j].y - bmin.y) / extent.y;
					boxdata.qupper[2] = (tUpper[j].z - bmin.z) / extent.z;

					q.boxData[j] = boxdata;
				}

				curVec.push_back(q);
			}

			std::sort(curVec.begin(), curVec.end());

			for (int i = 0; i < curVec.size(); i++) {
				for (int j = 0; j < 4; j++) {
					if ((curVec[i].childFlag & (1 << j)) == (1 << j)) break; // skip null node
					curVec[i].childs[j].boxIdx = (curVec.size() - i) + curVec[i].childs[j].boxIdx;
				}
			}

			res.insert(res.begin(), curVec.begin(), curVec.end());
		}
	}
	
	std::cout << "tree depth : " << treeDepth << std::endl;

	return res;
}