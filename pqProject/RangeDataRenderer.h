#pragma once
#include "Renderer.h"
#include "RangeDataReader.h"
#include "ColorUtil.h"

namespace AISTER_GRAPHICS_ENGINE {

	class Range_Renderer : public Renderer {
		std::vector<unsigned int> VBO, VAO;
		glm::vec4 _color;

		range_data* rd;

	public:
		void setShader(range_data* _rd, Shader* _shader) {
			rd = _rd;

			_color = glm::vec4(1, 0, 0, 0.2f);

			shader = _shader;

			initShader();
		}

		void initShader() {
			VBO = std::vector<unsigned int>(rd->datas.size());
			VAO = std::vector<unsigned int>(rd->datas.size());
			for (int i = 0; i < rd->datas.size(); i++) {
				glGenVertexArrays(1, &VAO[i]);
				glGenBuffers(1, &VBO[i]);
				glBindVertexArray(VAO[i]);

				glBindBuffer(GL_ARRAY_BUFFER, VBO[i]);
				glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * rd->datas[i].vertices.size(), &rd->datas[i].vertices[0], GL_STATIC_DRAW);

				glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
				glEnableVertexAttribArray(0);
			}
		}

		void Draw(Camera cam) {
			shader->call();

			for (int i = 0; i < rd->datas.size(); i++) {
				glm::mat4 trs = rd->datas[0].getTRS();
				glm::mat4 MVPmat = rd->camInfo[i].getProjectionMatrix() * rd->camInfo[i].getViewMatrix() * trs;

				_color = ColorUtil::getColorfromJET(i, 0, rd->datas.size());
				_color.a = 0.2f;

				GLuint location_MVP = glGetUniformLocation(shader->shaderProgram, "MVP");
				glUniformMatrix4fv(location_MVP, 1, GL_FALSE, &MVPmat[0][0]);
				GLuint location_color = glGetUniformLocation(shader->shaderProgram, "mtlColor");
				glUniform4fv(location_color, 1, &(_color[0]));

				glBindVertexArray(VAO[i]);
				glDrawArrays(GL_POINTS, 0, rd->datas[i].vertices.size());
			}

		}
	};

}