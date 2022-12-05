#pragma once

#include <iomanip>

#include "delaunay/gDel3D/GpuDelaunay.h"

#include "delaunay/DelaunayChecker.h"
#include "delaunay/InputCreator.h"

#include <string>
#include <fstream>
#include <vector>

#include <algorithm>

template <class T>
void endswap(T* objp);

float ReverseFloat(const float inFloat);
void binaryPrint(float* val);

typedef std::vector<float> color3f;

class Gdel3DParser {
public:
    GpuDel triangulator;
    Point3HVec   pointVec;
	std::vector<color3f> color;
	float min[3] = { 987654321.f,987654321.f, 987654321.f };
	float max[3] = { -987654321.f, -987654321.f , -987654321.f };
	int vecsize;
    GDelOutput   output;
    const int deviceIdx = 0;

	Gdel3DParser() {
		cudaSet();
	}

    void cudaSet() {
        CudaSafeCall(cudaSetDevice(deviceIdx));
        CudaSafeCall(cudaDeviceReset());
    }

    void compute() {
        triangulator.compute(pointVec, &output);
    }

    void summarize() {
        std::cout << std::endl;
        std::cout << "---- SUMMARY ----" << std::endl;
        std::cout << std::endl;

        std::cout << "PointNum       " << pointVec.size() << std::endl;
        std::cout << "FP Mode        " << ((sizeof(RealType) == 8) ? "Double" : "Single") << std::endl;
        std::cout << std::endl;
        std::cout << std::fixed << std::right << std::setprecision(2);
        std::cout << "TotalTime (ms) " << std::setw(10) << output.stats.totalTime << std::endl;
        std::cout << "InitTime       " << std::setw(10) << output.stats.initTime << std::endl;
        std::cout << "SplitTime      " << std::setw(10) << output.stats.splitTime << std::endl;
        std::cout << "FlipTime       " << std::setw(10) << output.stats.flipTime << std::endl;
        std::cout << "RelocateTime   " << std::setw(10) << output.stats.relocateTime << std::endl;
        std::cout << "SortTime       " << std::setw(10) << output.stats.sortTime << std::endl;
        std::cout << "OutTime        " << std::setw(10) << output.stats.outTime << std::endl;
        std::cout << "SplayingTime   " << std::setw(10) << output.stats.splayingTime << std::endl;
        std::cout << std::endl;									
        std::cout << "# Flips        " << std::setw(10) << output.stats.totalFlipNum << std::endl;
        std::cout << "# Failed verts " << std::setw(10) << output.stats.failVertNum << std::endl;
        std::cout << "# Final stars  " << std::setw(10) << output.stats.finalStarNum << std::endl;
    }


	void read(std::string filename) {
		std::ifstream plyFile(filename, std::ios::binary);

		std::string ss;

		plyFile >> ss;

		if (!ss._Equal("ply")) {
			std::cout << "file error" << std::endl;
			exit(0);
		}

		while (!plyFile.eof()) {
			plyFile >> ss;

			bool isEnd = false;

			if (ss._Equal("element")) {
				plyFile >> ss;

				if (ss._Equal("vertex")) {
					plyFile >> ss;

					vecsize = std::stoi(ss);
				}
			}

			if (ss._Equal("end_header")) {
				char space;
				plyFile.read(reinterpret_cast<char*>(&space), sizeof(char));

				for (int i = 0; i < vecsize; i++) {
					float x, y, z;

					plyFile.read((char*)(&x), sizeof(float));
					plyFile.read(reinterpret_cast<char*>(&y), sizeof(float));
					plyFile.read(reinterpret_cast<char*>(&z), sizeof(float));

					/*binaryPrint(&x); printf(" ");
					binaryPrint(&y); printf(" ");
					binaryPrint(&z); printf("\n");*/

					/*x = ReverseFloat(x);
					y = ReverseFloat(y);
					z = ReverseFloat(z);*/

					Point3 xyz = { x, y, z };
					pointVec.push_back(xyz);

					unsigned char r, g, b;
					plyFile.read(reinterpret_cast<char*>(&r), sizeof(char));
					plyFile.read(reinterpret_cast<char*>(&g), sizeof(char));
					plyFile.read(reinterpret_cast<char*>(&b), sizeof(char));

					color3f rgb = { r / 255.f,g / 255.f,b / 255.f };
					color.push_back(rgb);

					float nx, ny, nz;
					plyFile.read(reinterpret_cast<char*>(&nx), sizeof(float));
					plyFile.read(reinterpret_cast<char*>(&ny), sizeof(float));
					plyFile.read(reinterpret_cast<char*>(&nz), sizeof(float));

					min[0] = std::min(min[0], x);
					min[1] = std::min(min[1], y);
					min[2] = std::min(min[2], z);

					max[0] = std::max(max[0], x);
					max[1] = std::max(max[1], y);
					max[2] = std::max(max[2], z);
				}

				isEnd = true;
			}

			if (isEnd) break;
		}
	}
};


template <class T>
void endswap(T* objp)
{
	unsigned char* memp = reinterpret_cast<unsigned char*>(objp);
	std::reverse(memp, memp + sizeof(T));
}

float ReverseFloat(const float inFloat)
{
	float retVal;
	char* floatToConvert = (char*)&inFloat;
	char* returnFloat = (char*)&retVal;

	// swap the bytes into a temporary buffer
	returnFloat[0] = floatToConvert[3];
	returnFloat[1] = floatToConvert[2];
	returnFloat[2] = floatToConvert[1];
	returnFloat[3] = floatToConvert[0];

	return retVal;
}

void binaryPrint(float* val) {
	unsigned char ch[4];
	memcpy(ch, val, sizeof(float));

	printf("%x%x%x%x", ch[3], ch[2], ch[1], ch[0]);
}