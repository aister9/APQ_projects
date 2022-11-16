#pragma once
#include "Object.h"
#include "Camera.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

namespace AISTER_GRAPHICS_ENGINE {

	class geometry_range_data : public Object_t {
	public:
		std::vector<glm::vec3> vertices;

		void print() {
			std::cout << "Vertex size : " << vertices.size() << std::endl;
			std::cout << "Postion : " << position.x << ", " << position.y << ", " << position.z << std::endl;
			std::cout << "Rotation: " << rotation.w << ", " << rotation.x << ", " << rotation.y << ", " << rotation.z << std::endl;
			std::cout << "Scale: " << scale.x << ", " << scale.y << ", " << scale.z << std::endl;
		}
	};

	class range_data {
	public:
		std::vector<geometry_range_data> datas;
		std::vector<Camera> camInfo;

		Camera baseCam;

		void read(std::string folderPath, std::string confFileName) {
			std::ifstream m_file(folderPath + confFileName);

			while (!m_file.eof()) {
				std::string tmp_s;
				m_file >> tmp_s;

				if (tmp_s._Equal("camera")) {
					float posquat[7];
					for (int i = 0; i < 7; i++) {
						std::string tmp;
						m_file >> tmp;
						posquat[i] = std::stof(tmp);
					}
					baseCam.position = glm::vec3(posquat[0], posquat[1], posquat[2]);
					baseCam.rotation = glm::quat(posquat[3], posquat[4], posquat[5], posquat[6]);
				}
				else if (tmp_s._Equal("bmesh")) {
					m_file >> tmp_s;

					auto data = readPly(folderPath + tmp_s);

					float posquat[7];
					for (int i = 0; i < 7; i++) {
						std::string tmp;
						m_file >> tmp;
						posquat[i] = std::stof(tmp);
					}

					Camera cams;
					cams.position = glm::vec3(posquat[0], posquat[1], posquat[2]);
					cams.rotation = glm::quat(posquat[3], posquat[4], posquat[5], posquat[6]);

					glm::vec3 t = cams.getTRS() * glm::vec4(baseCam.position,1);
					glm::vec3 dir_t = glm::normalize(cams.getTRS() * baseCam.getTRS() * glm::vec4(cams.direction, 1));

					cams.position = t;
					cams.direction = dir_t;

					datas.push_back(data);
					camInfo.push_back(cams);
				}
			}

			m_file.close();
		}

		void setCameraAllResol(glm::vec2 resol) {
			for (auto& c : camInfo) {
				c.screenResolution = resol;
			}
		}

		void print() {
			for (auto& d : datas) {
				d.print();
			}
		}

		geometry_range_data readPly(std::string filePath) {
			std::ifstream m_file(filePath);
			float vertexsize = 0;

			geometry_range_data rg;

			while (!m_file.eof()) {
				std::string tmp;
				m_file >> tmp;

				if (tmp._Equal("element")) {
					m_file >> tmp;
					if (tmp._Equal("vertex")) {
						m_file >> tmp;
						vertexsize = atof(tmp.c_str());
					}
				}

				if (tmp._Equal("end_header")) {
					for (int i = 0; i < vertexsize; i++) {
						float x, y, z;
						m_file >> tmp;
						x = atof(tmp.c_str());
						m_file >> tmp;
						y = atof(tmp.c_str());
						m_file >> tmp;
						z = atof(tmp.c_str());

						glm::vec3 pos(x, y, z);

						rg.vertices.push_back(pos);
					}
				}
			}

			m_file.close();

			return rg;
		}
	};

}