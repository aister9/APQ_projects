#pragma once
#include "glHeaders.h"

namespace AISTER_GRAPHICS_ENGINE {
	class ColorUtil {
	public:

        static glm::vec4 getColorfromJET(float v, float vmin, float vmax) {
            glm::vec3 c = { 1.0, 1.0, 1.0 };
            float dv;

            if (v < vmin) v = vmin;
            if (v > vmax) v = vmax;
            dv = vmax - vmin;

            if (v < (vmin + 0.25 * dv)) {
                c.r = 0;
                c.g = 4 * (v - vmin) / dv;
            }
            else if (v < (vmin + 0.5 * dv)) {
                c.r = 0;
                c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
            }
            else if (v < (vmin + 0.75 * dv)) {
                c.r = 4 * (v - vmin - 0.5 * dv) / dv;
                c.b = 0;
            }
            else {
                c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
                c.b = 0;
            }
            return glm::vec4(c, 1.0f);
        }
	};
}