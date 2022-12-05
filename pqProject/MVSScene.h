#pragma once
#include "glHeaders.h"
#include "Camera.h"

const int tetTriangleOrder[] = {
	1,3,2,
	0,2,3,
	0,3,1,
	0,1,2
};

typedef struct Tet_t { int _v[4]; };
typedef struct Triangle_t {
	int _v[3];

	bool operator==(const Triangle_t &out) {
		if(_v[0] == out._v[0] || _v[0] == out._v[1] || _v[0] == out._v[2])
			if (_v[1] == out._v[0] || _v[1] == out._v[1] || _v[1] == out._v[2])
				if (_v[2] == out._v[0] || _v[2] == out._v[1] || _v[2] == out._v[2]) {
					return true;
				}
		return false;
	}
};
typedef struct TetraInfo {
	int _t[2];
};

typedef struct Triangle_SOA {
	std::vector<Triangle_t> vIdx;
	std::vector<float> weight;
	std::vector<TetraInfo> tIdx;
};


class MVS_PCD : public AISTER_GRAPHICS_ENGINE::Object_t {
public:

	AISTER_GRAPHICS_ENGINE::Camera cams[11];

	std::vector<glm::vec3> pts;
	std::vector<Tet_t> tet;

	Triangle_SOA faces;

	float minw;
	float maxw;

	//embedded cam info
	MVS_PCD(){
		int width = 2832, height = 2128, focal_length = 2987.018f;
		
		for (int i = 0; i < 11; i++) {
			cams[i].setFOV_K(focal_length, focal_length, width, height, 0, 1000);
		}

		cams[0].position = glm::vec3(0.025730884620962389,
			0.005601572570405825,
			0.01597567841113321);

		cams[1].position = glm::vec3(0.3810877798322228,
			-0.01665180729052384,
			-0.06503797152529445);

		cams[2].position = glm::vec3(0.6330633126858081,
			-0.021669968222753703,
			-0.0833297799710703);

		cams[3].position = glm::vec3(0.7705729788683636,
			-0.014459294101333682,
			-0.0391654577887085);

		cams[4].position = glm::vec3(1.0044853197972546,
			0.0013310199727547476,
			0.03379707902389767);

		cams[5].position = glm::vec3(1.1992924274816506,
			0.026952636241720668,
			0.1497150231681726);

		cams[6].position = glm::vec3(1.3367036438812217,
			0.06544004054031413,
			0.3179096555741522);

		cams[7].position = glm::vec3(1.3896499687559494,
			0.11932044219851415,
			0.5746557540157031);

		cams[8].position = glm::vec3(1.4397271231838996,
			0.18190199911482564,
			0.8435911360770654);

		cams[9].position = glm::vec3(1.45543861907246,
			0.2344942332385428,
			1.083227922032751);

		cams[10].position = glm::vec3(1.3856453288653988,
			0.28775204994020478,
			1.3290640112599066);



		glm::mat3x3 rot =
			(glm::mat3x3(
				glm::vec3(0.9998209035428125,
					0.013495867621449609,
					0.013267343214770324),
				glm::vec3(-0.013393139173710739,
					0.9998798725778282,
					-0.007801553487585845),
				glm::vec3(-0.013371038176140958,
					0.007622464882854893,
					0.9998815496683604)
			));
		cams[0].rotation = glm::toQuat(rot);

		cams[0]._near = 0.0003f;

		cams[0].direction = glm::normalize(glm::toMat3(cams[0].rotation) * glm::vec3(0, 0, -1));

	}

	void read(std::string path) {
		std::ifstream ifile(path);

		std::string ss;
		while (!ifile.eof()) {
			ifile >> ss;

			if (ss._Equal("xyz")) {
				float x, y, z;
				ifile >> ss;
				x = stof(ss);
				ifile >> ss;
				y = stof(ss);
				ifile >> ss;
				z = stof(ss);

				pts.push_back(glm::vec3(x, y, z));
			}
			if (ss._Equal("tet")) {
				float v0, v1, v2, v3;
				ifile >> ss;
				v0 = stof(ss);
				ifile >> ss;
				v1 = stof(ss);
				ifile >> ss;
				v2 = stof(ss);
				ifile >> ss;
				v3 = stof(ss);

				Tet_t t;
				t._v[0] = v0;
				t._v[1] = v1;
				t._v[2] = v2;
				t._v[3] = v3;

				tet.push_back(t);
			}
		}

		ifile.close();

		buildFace();
	}

	AISTER_GRAPHICS_ENGINE::bbox get_3d_bbox() {
		glm::vec3 _min(pts[0] * scale);
		glm::vec3 _max(pts[0] * scale);

		for (auto v : pts) {
			_min = glm::min(_min, v * scale);
			_max = glm::max(_max, v * scale);
		}

		AISTER_GRAPHICS_ENGINE::bbox res;
		res.center = (_min + _max) * 0.5f;
		res.size = (_max - _min);

		return res;
	}


	void readPrecomputed(std::string path) {
		std::ifstream ifile(path);

		std::string ss;
		while (!ifile.eof()) {
			ifile >> ss;

			if (ss._Equal("xyz")) {
				float x, y, z;
				ifile >> ss;
				x = stof(ss);
				ifile >> ss;
				y = stof(ss);
				ifile >> ss;
				z = stof(ss);

				pts.push_back(glm::vec3(x, y, z));
			}
			if (ss._Equal("tri")) {
				float v0, v1, v2;
				ifile >> ss;
				v0 = stoi(ss);
				ifile >> ss;
				v1 = stoi(ss);
				ifile >> ss;
				v2 = stoi(ss);
				Triangle_t t;

				t._v[0] = v0;
				t._v[1] = v1;
				t._v[2] = v2;

				faces.vIdx.push_back(t);
			}
			if (ss._Equal("w")) {
				float v0;
				ifile >> ss;
				v0 = stof(ss);

				faces.weight.push_back(v0);
			}
			if (ss._Equal("tet")) {
				float v0, v1;
				ifile >> ss;
				v0 = stoi(ss);
				ifile >> ss;
				v1 = stoi(ss);

				TetraInfo t;
				t._t[0] = v0;
				t._t[1] = v1;

				faces.tIdx.push_back(t);
			}
		}

		ifile.close();
	}

	void buildFace() {
		std::cout << "Build Faces " << std::endl;
		auto _start = glfwGetTime();
		faces.vIdx.resize(tet.size()*4);
		faces.tIdx.resize(tet.size() * 4);
		faces.weight.resize(tet.size() * 4);
		unsigned int pos = 0;
		for (int i = 0; i < tet.size(); i++) {
			for (int j = 0; j < 4; j++) {
				Triangle_t t;
				t._v[0] = tet[i]._v[tetTriangleOrder[j * 3 + 0]];
				t._v[1] = tet[i]._v[tetTriangleOrder[j * 3 + 1]];
				t._v[2] = tet[i]._v[tetTriangleOrder[j * 3 + 2]];

				bool isFind = false;
				for (int k = 0; k < pos; k++) {
					if (faces.vIdx[k] == t) {
						isFind = true;
						faces.tIdx[k]._t[1] = i;
						break;
					}
				}
				if (!isFind)
				{
					faces.vIdx[pos] = t;
					faces.weight[pos] = 0;
					TetraInfo info;
					info._t[0] = i;
					info._t[1] = -1;
					faces.tIdx[pos] = info;
					pos++;
				}
			}
			if (i % (tet.size() / 100) == 0) {
				std::cout << i / (tet.size() / 100) << "% complete" << std::endl;
			}
		}
		auto _end = glfwGetTime();

		faces.vIdx.resize(pos);
		faces.tIdx.resize(pos);
		faces.weight.resize(pos);

		std::cout << "Face Build Time(ms) : " << (_end - _start) * 1000 << std::endl;
	}

	void saveAs(std::string fileName) {
		std::ofstream os(fileName);

		if (!os.is_open()) return;

		os << "vertex " << pts.size() << std::endl;
		for (int i = 0; i < pts.size(); i++) {
			os << "xyz " << pts[i].x << " " << pts[i].y << " " << pts[i].z << std::endl;
		}
		os << "Triangle " << faces.vIdx.size() << std::endl;
		for (int i = 0; i < faces.vIdx.size(); i++) {
			os << "tri " << faces.vIdx[i]._v[0] << " " << faces.vIdx[i]._v[1] << " " << faces.vIdx[i]._v[2] << std::endl;
			os << "w " << faces.weight[i] << std::endl;
			os << "tet " << faces.tIdx[i]._t[0] << " " << faces.tIdx[i]._t[1] << std::endl;
		}

		os.close();
	}

	std::pair<float, float> getMinMaxWeight() {
		float min = faces.weight[0], max = faces.weight[0];
		for (auto& w : faces.weight) {
			min = std::min(min, w);
			max = std::max(max, w);
		}

		minw = min;
		maxw = max;

		std::cout << minw << std::endl;
		std::cout << maxw << std::endl;


		return std::make_pair(min, max);
	}
};


namespace AISTER_GRAPHICS_ENGINE {

	class PCDRenderer : public Renderer {
		unsigned int VBO, VAO, EBO, CBO;
		glm::vec4 _color;

		MVS_PCD* rd;

	public:
		void setShader(MVS_PCD* _rd, Shader* _shader) {
			rd = _rd;

			_color = glm::vec4(1, 0, 0, 0.2f);

			shader = _shader;

			initShader();
		}

		void initShader() {
			printf("call shader");
			glGenVertexArrays(1, &VAO);
			glGenBuffers(1, &VBO);
			glGenBuffers(1, &EBO);
			glGenBuffers(1, &CBO);
			glBindVertexArray(VAO);

			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * rd->pts.size(), &rd->pts[0], GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * 3 * rd->faces.vIdx.size(), &rd->faces.vIdx[0], GL_STATIC_DRAW);

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			glBindBuffer(GL_ARRAY_BUFFER, CBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * rd->faces.weight.size(), &rd->faces.weight[0], GL_STATIC_DRAW);
			glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(1);
		}

		void Draw(int camIdx) {
			shader->call();
			glm::mat4 trs = rd->getTRS();
			//glm::mat4 MVPmat = cam.getProjectionMatrix() *cam.getViewMatrix() * trs;
			glm::mat4 MVPmat = rd->cams[camIdx].getProjectionMatrix() * rd->cams[camIdx].getViewMatrix() * trs;

			GLuint location_MVP = glGetUniformLocation(shader->shaderProgram, "MVP");
			glUniformMatrix4fv(location_MVP, 1, GL_FALSE, &MVPmat[0][0]);
			GLuint location_color = glGetUniformLocation(shader->shaderProgram, "mtlColor");
			glUniform4fv(location_color, 1, &(_color[0]));

			glBindVertexArray(VAO);
			glDrawArrays(GL_POINTS, 0, rd->pts.size());
		}

		void Draw(Camera cam) {
			shader->call();
			glm::mat4 trs = rd->getTRS();
			glm::mat4 MVPmat = cam.getProjectionMatrix() *cam.getViewMatrix() * trs;
			//glm::mat4 MVPmat = rd->cams[camIdx].getProjectionMatrix() * rd->cams[camIdx].getViewMatrix() * trs;

			GLuint location_MVP = glGetUniformLocation(shader->shaderProgram, "MVP");
			glUniformMatrix4fv(location_MVP, 1, GL_FALSE, &MVPmat[0][0]);
			GLuint location_color = glGetUniformLocation(shader->shaderProgram, "mtlColor");
			glUniform4fv(location_color, 1, &(_color[0]));

			glBindVertexArray(VAO);
			glDrawArrays(GL_POINTS, 0, rd->pts.size());
		}

		void DrawTetra(Camera cam) {
			shader->call();
			glm::mat4 trs = rd->getTRS();
			glm::mat4 MVPmat = cam.getProjectionMatrix() * cam.getViewMatrix() * trs;
			//glm::mat4 MVPmat = rd->cams[camIdx].getProjectionMatrix() * rd->cams[camIdx].getViewMatrix() * trs;

			GLuint location_MVP = glGetUniformLocation(shader->shaderProgram, "MVP");
			glUniformMatrix4fv(location_MVP, 1, GL_FALSE, &MVPmat[0][0]);
			GLuint location_color = glGetUniformLocation(shader->shaderProgram, "mtlColor");
			glUniform4fv(location_color, 1, &(_color[0]));

			GLuint loc_min_w = glGetUniformLocation(shader->shaderProgram, "minweight");
			glUniform1f(loc_min_w, rd->minw);

			GLuint loc_max_w = glGetUniformLocation(shader->shaderProgram, "maxweight");
			glUniform1f(loc_max_w, rd->maxw);

			glBindVertexArray(VAO);
			glDrawElements(GL_TRIANGLES, rd->faces.vIdx.size() * 3, GL_UNSIGNED_INT, 0);
		}
	};

}