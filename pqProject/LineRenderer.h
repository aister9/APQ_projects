#pragma once
#include "glHeaders.h"
#include "Renderer.h"

namespace AISTER_GRAPHICS_ENGINE {
	class LineRenderer : public Renderer {
		unsigned int VBO, VAO;
		glm::vec3 startend[2];
		glm::vec4 lineColor;

	public:
		void setShaderLine(glm::vec3 start, glm::vec3 end, Shader* _shader) {
			startend[0] = start;
			startend[1] = end;

			lineColor = glm::vec4(1, 0, 0, 1);

			shader = _shader;

			initShader();
		}

		void initShader() {
			glGenVertexArrays(1, &VAO);
			glGenBuffers(1, &VBO);
			glBindVertexArray(VAO);

			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * 2, &startend[0], GL_STATIC_DRAW);

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(0);
		}

		void setColor(glm::vec4 color) {
			lineColor = color;
		}

		void Draw(Camera cam) {
			shader->call();
			glm::mat4 MVPmat = cam.getProjectionMatrix() * cam.getViewMatrix(glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

			GLuint location_MVP = glGetUniformLocation(shader->shaderProgram, "MVP");
			glUniformMatrix4fv(location_MVP, 1, GL_FALSE, &MVPmat[0][0]);
			GLuint location_color = glGetUniformLocation(shader->shaderProgram, "mtlColor");
			glUniform4fv(location_color, 1, &(lineColor[0]));

			glBindVertexArray(VAO);
			glDrawArrays(GL_LINES, 0, 2);
		}
	};
}