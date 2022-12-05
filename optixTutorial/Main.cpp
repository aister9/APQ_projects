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
            //sample.setCamera(camera);
        }

        virtual void render() override
        {
            /*if (cameraFrame.modified) {
                sample.setCamera(Camera{ cameraFrame.get_from(),
                                         cameraFrame.get_at(),
                                         cameraFrame.get_up() });
                cameraFrame.modified = false;
            }*/
            sample.setInitDistance(2.0f);
            sample.setWeightBuffer(sizeof(float)* weights.size());
            sample.render();
        }

        virtual void draw() override
        {
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
                        continue;
                        ofs << "miss ";
                    }
                    else {
                        ofs << "hit ";
                    }
                    ofs << rayOrigin[i].x << " " << rayOrigin[i].y << " " << rayOrigin[i].z << " to ";
                    ofs << rayTarget[i].x << " " << rayTarget[i].y << " " << rayTarget[i].z << "\n";

                }
            }

            std::cout << "complete" <<std::endl;

            ofs.close();
        }

        vec2i                 fbSize;
        GLuint                fbTexture{ 0 };
        SampleRenderer        sample;
        std::vector<uint32_t> pixels;
        std::vector<vec3f> rayOrigin;
        std::vector<vec3f> rayTarget;

        public:
            std::vector<float> weights;
    };


    /*! main entry point to this example - initially optix, print hello
      world, then exit */
    extern "C" int main(int ac, char** av)
    {
        try {
            std::vector<float> weights;
            std::vector<std::pair<int, int>> tetras;
            Model* model = loadAPQ("models/tetrahedron.apq", weights, tetras);

            std::cout << "model vertex size : " << model->meshes[0]->vertex.size() << std::endl;
            std::cout << "model tri size : " << model->meshes[0]->index.size() << std::endl;
            std::cout << "model weight size : " << weights.size() << std::endl;
            std::cout << "model tetras size : " << tetras.size() << std::endl;
              
            vec3f from = model->bounds.center() + model->bounds.size();
            vec3f at = model->bounds.center();

            Camera camera = { /*from*/from,
                /* at */at ,
                /* up */vec3f(0.f,1.f,0.f) };
            // something approximating the scale of the world, so the
            // camera knows how much to move for any given user interaction:
            const float worldScale = length(model->bounds.span());

            vec3f cams[11];

            cams[0] = vec3f(0.025730884620962389,
                0.005601572570405825,
                0.01597567841113321);

            cams[1] = vec3f(0.3810877798322228,
                -0.01665180729052384,
                -0.06503797152529445);

            cams[2] = vec3f(0.6330633126858081,
                -0.021669968222753703,
                -0.0833297799710703);

            cams[3] = vec3f(0.7705729788683636,
                -0.014459294101333682,
                -0.0391654577887085);

            cams[4] = vec3f(1.0044853197972546,
                0.0013310199727547476,
                0.03379707902389767);

            cams[5] = vec3f(1.1992924274816506,
                0.026952636241720668,
                0.1497150231681726);

            cams[6] = vec3f(1.3367036438812217,
                0.06544004054031413,
                0.3179096555741522);

            cams[7] = vec3f(1.3896499687559494,
                0.11932044219851415,
                0.5746557540157031);

            cams[8] = vec3f(1.4397271231838996,
                0.18190199911482564,
                0.8435911360770654);

            cams[9] = vec3f(1.45543861907246,
                0.2344942332385428,
                1.083227922032751);

            cams[10] = vec3f(1.3856453288653988,
                0.28775204994020478,
                1.3290640112599066);

            SampleRenderer sample(model);
            sample.setCamera(cams);
            sample.setVertexList(model);
            sample.resize(vec2i(model->meshes[0]->vertex.size(), 11));
            sample.setInitDistance(4.5f);
            sample.setWeightBuffer(sizeof(float) * weights.size());
            sample.render();
            sample.downloadWeightResult(weights.data(), weights.size());

            std::cout << "save to apq" << std::endl;
            saveAPQ(model, "tet.apq", weights, tetras);


            //SampleWindow* window = new SampleWindow("Optix 7 Course Example",
            //    model, camera, worldScale);
            //window->weights = weights;
            //window->run();
            //window->parseRayResult("test.txt");
            //window->sample.downloadWeightResult(weights.data(), weights.size());

            //for (int i = 0; i < weights.size(); i++) {

            //    if (weights[i] != 0) {
            //        std::cout << i << " : " << weights[i] << std::endl;
            //    }
            //}

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
