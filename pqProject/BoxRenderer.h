#pragma once
#include "Renderer.h"
#include "BoundingBox.h"
#include <vector>

namespace AISTER_GRAPHICS_ENGINE {

	const int boxIndices[]{
		0,1,2, 1,2,3, // front
		4,5,6, 5,6,7, // back

		0,1,4, 1,4,5, // bottom
		2,3,6, 3,6,7, // top

		0,4,2, 4,2,6, // left
		1,5,3, 5,3,7 // right
	};

	class BoxRenderer : public Renderer {
		unsigned int VBO, VAO, EBO;
		glm::vec4 boxColor;
		glm::vec4 edgeColor;
		float thickness;

		bbox* box;
		std::vector<glm::vec3> verticies;
		std::vector<int> indicies;
	public:
		void setShaderBox(bbox* _box, Shader* _shader) {
			box = _box;
			verticies = box->getCorner();
			indicies = std::vector<int>{
					0, 1, 2, 1, 2, 3, // front
					4, 5, 6, 5, 6, 7, // back

					0, 1, 4, 1, 4, 5, // bottom
					2, 3, 6, 3, 6, 7, // top

					0, 4, 2, 4, 2, 6, // left
					1, 5, 3, 5, 3, 7 // right
			};

			boxColor = glm::vec4(1, 1, 1, 0.2f);
			edgeColor = glm::vec4(1, 0, 0, 1);
			thickness = 0.1f;

			shader = _shader;

			initShader();
		}

		void setEdgeColor(glm::vec4 c) {
			edgeColor = c;
		}

		void initShader() {
			glGenVertexArrays(1, &(this->VAO));
			glGenBuffers(1, &(this->VBO));
			glGenBuffers(1, &(this->EBO));
			glBindVertexArray(this->VAO);

			glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * verticies.size(), &verticies[0], GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * indicies.size(), &indicies[0], GL_STATIC_DRAW);

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
		}

		void Draw(Camera cam) {
			shader->call();
			glm::mat4 MVPmat = cam.getProjectionMatrix() * cam.getViewMatrix(glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

			GLuint location_MVP = glGetUniformLocation(shader->shaderProgram, "MVP");
			glUniformMatrix4fv(location_MVP, 1, GL_FALSE, &MVPmat[0][0]);
			GLuint location_color = glGetUniformLocation(shader->shaderProgram, "mtlColor");
			glUniform4fv(location_color, 1, &(boxColor[0]));
			glUniform3fv(glGetUniformLocation(shader->shaderProgram, "_center"), 1, &(box->center[0]));
			glUniform3fv(glGetUniformLocation(shader->shaderProgram, "_size"), 1, &(box->size[0]));
			glUniform4fv(glGetUniformLocation(shader->shaderProgram, "_edgeColor"), 1, &(edgeColor[0]));
			glUniform1f(glGetUniformLocation(shader->shaderProgram, "_thickNess"), thickness);

			glBindVertexArray(this->VAO);
			glDrawElements(GL_TRIANGLES, indicies.size(), GL_UNSIGNED_INT, 0);
		}
	};
}