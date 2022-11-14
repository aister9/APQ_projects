#pragma once
#include "glHeaders.h"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

namespace AISTER_GRAPHICS_ENGINE {
	class Texture {
	public:
		int width;
		int height;
		int type;
		uchar* data;

		unsigned int texture;

		Texture(std::string filePath) {
			cv::Mat image = cv::imread(filePath, cv::IMREAD_UNCHANGED);

			cv::cvtColor(image, image, cv::COLOR_BGRA2RGBA);
			cv::flip(image, image, 0);

			width = image.cols;
			height = image.rows;

			data = image.data;
			type = image.type();

			glGenTextures(1, &texture);
			glBindTexture(GL_TEXTURE_2D, texture);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
			glGenerateMipmap(GL_TEXTURE_2D);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		}
	};
}