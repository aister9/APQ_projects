#pragma once
#include "plyData.h"
#include "Shader.h"
#include "Camera.h"
#include "Texture.h"

namespace AISTER_GRAPHICS_ENGINE {
	class Renderer {
	public:
		Shader* shader;
		void Draw();
	};


	class plyRenderer : public Renderer{
		unsigned int VBO, VAO, EBO, UVBO;
		PLYdata* data;
		Texture* textures;

	public:
		void SetShaderPLY(PLYdata *_data, Shader* _shader, Texture *_textures) {
			data = _data;
			shader = _shader;
			textures = _textures;

			initShader();
		}

		PLYdata* getPlyDataPtr() { return data; }

		void initShader() {
			glGenVertexArrays(1, &(this->VAO));
			glGenBuffers(1, &(this->VBO));
			glGenBuffers(1, &(this->EBO));
			glGenBuffers(1, &(this->UVBO));
			glBindVertexArray(this->VAO);

			glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * data->vertices.size(), &data->vertices[0], GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * data->faces.size(), &data->faces[0], GL_STATIC_DRAW);

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			glBindBuffer(GL_ARRAY_BUFFER, this->UVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * data->uvs.size(), &data->uvs[0], GL_STATIC_DRAW);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(1);
		}

		void Draw(Camera cam, bool depthOption = false) {
			shader->call();

			glm::mat4 trs = data->getTRS();
			//glm::mat4 MVPmat = cam.getProjectionMatrix() * cam.getViewMatrix(glm::vec3(0, 0, 0), glm::vec3(0, 1, 0)) * trs;
			glm::mat4 MVPmat = cam.getProjectionMatrix() * cam.getViewMatrix() * trs;

			GLuint location_MVP = glGetUniformLocation(shader->shaderProgram, "MVP");
			glUniformMatrix4fv(location_MVP, 1, GL_FALSE, &MVPmat[0][0]);
			GLuint location_isDepth = glGetUniformLocation(shader->shaderProgram, "drawDepth");
			glUniform1i(location_isDepth, depthOption);

			glBindTexture(GL_TEXTURE_2D, textures->texture);
			glBindVertexArray(this->VAO);
			glDrawElements(GL_TRIANGLES, data->faces.size(), GL_UNSIGNED_INT, 0);
		}

		void Draw(Camera cam, glm::vec4 color, bool depthOption = false) {
			shader->call();

			glm::mat4 trs = data->getTRS();
			glm::mat4 MVPmat = cam.getProjectionMatrix() * cam.getViewMatrix(glm::vec3(0,0,0), glm::vec3(0,1,0)) * trs;

			GLuint location_MVP = glGetUniformLocation(shader->shaderProgram, "MVP");
			glUniformMatrix4fv(location_MVP, 1, GL_FALSE, &MVPmat[0][0]);
			GLuint location_color = glGetUniformLocation(shader->shaderProgram, "mtlColor");
			glUniform4fv(location_color, 1, &(color)[0]);
			GLuint location_isDepth = glGetUniformLocation(shader->shaderProgram, "drawDepth");
			glUniform1i(location_isDepth, depthOption);

			glBindTexture(GL_TEXTURE_2D, textures->texture);
			glBindVertexArray(this->VAO);
			glDrawElements(GL_TRIANGLES, data->faces.size(), GL_UNSIGNED_INT, 0);

		}

		void Draw(glm::mat4 MVP, glm::vec4 color) {
			shader->call();

			GLuint location_MVP = glGetUniformLocation(shader->shaderProgram, "MVP");
			glUniformMatrix4fv(location_MVP, 1, GL_FALSE, &MVP[0][0]);
			GLuint location_color = glGetUniformLocation(shader->shaderProgram, "mtlColor");
			glUniform4fv(location_color, 1, &(color)[0]);
			GLuint location_isDepth = glGetUniformLocation(shader->shaderProgram, "drawDepth");
			glUniform1i(location_isDepth, false);

			glBindTexture(GL_TEXTURE_2D, textures->texture);
			glBindVertexArray(this->VAO);
			glDrawElements(GL_TRIANGLES, data->faces.size(), GL_UNSIGNED_INT, 0);
		}
	};
}