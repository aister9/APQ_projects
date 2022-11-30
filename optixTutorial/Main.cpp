// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "SampleRenderer.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

#include <fstream>
#include "MyCudaAdd.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

    struct SampleWindow : public GLFCameraWindow
    {
        SampleWindow(const std::string& title,
            const Model* model,
            const Camera& camera,
            const float worldScale)
            : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale),
            sample(model)
        {
            sample.setCamera(camera);
        }

        virtual void render() override
        {
            if (cameraFrame.modified) {
                sample.setCamera(Camera{ cameraFrame.get_from(),
                                         cameraFrame.get_at(),
                                         cameraFrame.get_up() });
                cameraFrame.modified = false;
            }
            sample.render();
        }

        virtual void draw() override
        {
            sample.downloadPixels(pixels.data());
            sample.downloadRayResult(rayOrigin.data(), rayTarget.data());
            if (fbTexture == 0)
                glGenTextures(1, &fbTexture);

            glBindTexture(GL_TEXTURE_2D, fbTexture);
            GLenum texFormat = GL_RGBA;
            GLenum texelType = GL_UNSIGNED_BYTE;
            glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                texelType, pixels.data());

            glDisable(GL_LIGHTING);
            glColor3f(1, 1, 1);

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, fbTexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glDisable(GL_DEPTH_TEST);

            glViewport(0, 0, fbSize.x, fbSize.y);

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

            glBegin(GL_QUADS);
            {
                glTexCoord2f(0.f, 0.f);
                glVertex3f(0.f, 0.f, 0.f);

                glTexCoord2f(0.f, 1.f);
                glVertex3f(0.f, (float)fbSize.y, 0.f);

                glTexCoord2f(1.f, 1.f);
                glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

                glTexCoord2f(1.f, 0.f);
                glVertex3f((float)fbSize.x, 0.f, 0.f);
            }
            glEnd();
        }

        virtual void resize(const vec2i& newSize)
        {
            fbSize = newSize;
            sample.resize(newSize);
            pixels.resize(newSize.x * newSize.y);
            rayOrigin.resize(newSize.x * newSize.y);
            rayTarget.resize(newSize.x * newSize.y);
        }

        void parseRayResult(const std::string& output) {
            std::ofstream ofs(output);

            std::cout << "Save ray result ... ";

            if (ofs.is_open()) {
                ofs << "Resolution : " << fbSize.x << ", " << fbSize.y << std::endl;
                for (int i = 0; i < rayOrigin.size(); i++) {
                    if (rayTarget[i].x < -100000.f && rayTarget[i].y < -100000.f && rayTarget[i].z < -100000.f) {
                        ofs << "miss ";
                    }
                    else {
                        ofs << "hit ";
                    }
                    ofs << rayOrigin[i].x << " " << rayOrigin[i].y << " " << rayOrigin[i].z << " to ";
                    ofs << rayTarget[i].x << " " << rayTarget[i].y << " " << rayTarget[i].z << "\n";
                }
            }

            std::cout << "complete";

            ofs.close();
        }

        vec2i                 fbSize;
        GLuint                fbTexture{ 0 };
        SampleRenderer        sample;
        std::vector<uint32_t> pixels;
        std::vector<vec3f> rayOrigin;
        std::vector<vec3f> rayTarget;
    };


    /*! main entry point to this example - initially optix, print hello
      world, then exit */
    extern "C" int main(int ac, char** av)
    {
        try {
//            Model* model = loadOBJ(
//#ifdef _WIN32
//                // on windows, visual studio creates _two_ levels of build dir
//                // (x86/Release)
//                "models/sponza.obj"
//#else
//                // on linux, common practice is to have ONE level of build dir
//                // (say, <project>/build/)...
//                "/models/sponza.obj"
//#endif
//            );
            Model* model = loadPLY("models/Cherries.ply","models/Cherries.png");

            Camera camera = { /*from*/vec3f(30.f, 0.f, 0.f),
                /* at */vec3f(0.f,0.f,0.f) ,
                /* up */vec3f(0.f,1.f,0.f) };
            // something approximating the scale of the world, so the
            // camera knows how much to move for any given user interaction:
            const float worldScale = length(model->bounds.span());

            SampleWindow* window = new SampleWindow("Optix 7 Course Example",
                model, camera, worldScale);
            window->resize(vec2i(640, 480));
            window->run();
            window->parseRayResult("test.txt");
        }
        catch (std::runtime_error& e) {
            std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << GDT_TERMINAL_DEFAULT << std::endl;
            std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your optix7course/models directory?" << std::endl;
            exit(1);
        }
        return 0;
    }

} // ::osc
