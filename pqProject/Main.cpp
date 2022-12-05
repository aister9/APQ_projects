#include <iostream>

#include "glHeaders.h"
#include "plyData.h"
#include "Texture.h"
#include "Shader.h"
#include "Renderer.h"
#include "BoxRenderer.h"
#include "LineRenderer.h"
#include "RangeDataReader.h"
#include "RangeDataRenderer.h"
#include "Scene.h"
#include "QBVH4.h"
#include "Ray.h"

#include "MVSInterface.h"
#include "MVSScene.h"

using namespace std;

glm::vec3 camPos(30, 0, 0);

bool glfwewInit(GLFWwindow** window, int width, int height) {
    if (!glfwInit()) return false; // glfw 초기화

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    *(window) = glfwCreateWindow(width, height, "Test window", nullptr, nullptr);

    if (!window) {
        glfwTerminate();
        cout << "failed" << endl;
        return false;
    }

    glfwMakeContextCurrent(*window); // 윈도우 컨텍스트 생성

    glfwSwapInterval(0); // 스왑 간격 : 0 설정하면 fps 제한X, 1 설정하면 fps 제한 60

    if (glewInit() != GLEW_OK) { // GLEW 초기호 실패하면 종료
        glfwTerminate();
        return false;
    }

    cout << glGetString(GL_VERSION) << endl; // OpenGL 버전

    return true;
}

glm::vec4 getColorfromJET(float v, float vmin, float vmax) {
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

void parseRayResult(const std::string& output, glm::vec3 rayOrigin, std::vector<AISTER_GRAPHICS_ENGINE::RayHit> hitGroups, int width, int height) {
    std::ofstream ofs(output);

    std::cout << "Save ray result ... ";

    if (ofs.is_open()) {
        ofs << "Resolution : " << width << ", " << height << std::endl;
        for (int i = 0; i < hitGroups.size(); i++) {
            if (!hitGroups[i].isHit) {
                ofs << "miss ";
            }
            else {
                ofs << "hit ";
            }
            ofs << rayOrigin.x << " " << rayOrigin.y << " " << rayOrigin.z << " to ";
            ofs << hitGroups[i].position.x << " " << hitGroups[i].position.y << " " << hitGroups[i].position.z << "\n";
        }
    }

    std::cout << " complete" << std::endl;

    ofs.close();
}

std::vector<glm::vec3> getRayResult(const std::string& input) {
    std::ifstream ifs(input);

    std::vector<glm::vec3> output;

    while (!ifs.eof()) {
        std::string s;

        ifs >> s;
        if (s._Equal("to")) {
            glm::vec3 target;
            for (int i = 0; i < 3; i++) {
                ifs >> s;
                float vv = std::stof(s);
                target[i] = vv;
            }
            output.push_back(target);
        }
    }

    ifs.close();

    return output;
}

int main(int argc, char* args[]) {

    GLFWwindow* window;
    int width = 1200, height = 800;

    if (!glfwewInit(&window, width, height)) return -1;

    AISTER_GRAPHICS_ENGINE::PLYdata plys("example/Cherries.ply");
    plys.print();
    //plys.position = -1.f * plys.getCenter();
    auto boxs = plys.get_3d_bbox();

    cout << "r : " << plys.get_r_bbox() << endl;
    AISTER_GRAPHICS_ENGINE::Texture tex("example/Cherries.png");

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
        Rangedata test
    */
    AISTER_GRAPHICS_ENGINE::range_data rd;
    rd.read("example/data/", "bun.conf");
    rd.baseCam.direction = glm::normalize(glm::mat3(rd.baseCam.rotation) * glm::vec3(0, 0, -1));
    rd.setCameraAllResol(glm::vec2(width, height));


    AISTER_GRAPHICS_ENGINE::PLYdata Bunny("example/bun_zipper.ply", false);
    Bunny.print();


    AISTER_GRAPHICS_ENGINE::Shader range_shader;
    range_shader.initShaders("pc_v.glsl", "pc_f.glsl");
    AISTER_GRAPHICS_ENGINE::Range_Renderer rr;
    rr.setShader(&rd, &range_shader);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    AISTER_GRAPHICS_ENGINE::Shader sh;
    sh.initShaders("mesh_vertex.glsl", "mesh_frag.glsl");

    AISTER_GRAPHICS_ENGINE::plyRenderer renderer;
    renderer.SetShaderPLY(&plys, &sh, &tex);

    AISTER_GRAPHICS_ENGINE::plyRenderer bunrenderer;
    bunrenderer.SetShaderPLY(&Bunny, &sh, &tex);

    AISTER_GRAPHICS_ENGINE::Camera cam;
    cam.position = camPos;
    cam._far = 1000.f;
    
    cam.screenResolution = glm::vec2(width, height);

    cout << "camera matrix : " << glm::to_string(cam.getProjectionMatrix()) << endl;
    
    AISTER_GRAPHICS_ENGINE::QBVH4Node test;
    cout << "QBVH4Node Size : " << test.getSize_t() << endl;

    auto triList = build(plys.vertices, plys.faces, plys.getTRS());
    auto bvh = build(plys.vertices, triList, plys.getTRS());

    cout << "Triangle list length : " << triList.size() << endl;
    cout << "BVH list length : " << bvh.size() << endl;

    AISTER_GRAPHICS_ENGINE::Shader BoxShader;
    BoxShader.initShaders("box_v.glsl", "box_f.glsl");

    AISTER_GRAPHICS_ENGINE::bbox bboxs[64];
    for (int i = 0; i < 64; i++) {
        bboxs[i] = AISTER_GRAPHICS_ENGINE::bbox(bvh[157+i].getCenter(), bvh[157+i].getSize());
    }

    AISTER_GRAPHICS_ENGINE::BoxRenderer boxRenderer[64];
    for (int i = 0; i < 64; i++) {
        boxRenderer[i].setShaderBox(&bboxs[i], &BoxShader);
        boxRenderer[i].setEdgeColor(getColorfromJET(i, 0, 64));
    }

    AISTER_GRAPHICS_ENGINE::Shader lineShader;
    lineShader.initShaders("line_v.glsl", "line_f.glsl");


    vector<glm::vec3> dirList;

    for (int yy = 0; yy < height; yy++) {
        for (int xx = 0; xx < width; xx++) {
            glm::vec2 screen(glm::vec2(xx + .5f, yy + .5f)/cam.screenResolution);

            glm::vec3 rayDir = glm::normalize(cam.direction
                + (screen.x - 0.5f) * cam.getHorizontal()
                + (screen.y - 0.5f) * cam.getVertical());

            dirList.push_back(rayDir);
        }
    }

    vector<AISTER_GRAPHICS_ENGINE::RayHit> hitGroup;
    vector<int> hitInd;

    vector<glm::vec3> translated_vertices;
    for (auto v : plys.vertices) {
        glm::vec3 res = plys.getTRS()* glm::vec4(v, 1);

        translated_vertices.push_back(res);
    }

    auto _start = glfwGetTime();
    for (int i = 0; i < dirList.size(); i++) {
        AISTER_GRAPHICS_ENGINE::Ray ray(cam.position, dirList[i]);

        AISTER_GRAPHICS_ENGINE::RayHit res = ray.traverse(bvh, translated_vertices, triList);

        if (res.isHit) {
            hitInd.push_back(i);
            hitGroup.push_back(res);
        }
    }
    auto _end = glfwGetTime();

    auto _g_start = glfwGetTime();
    auto hitGroup2 = RayTraverse(cam.position, dirList, bvh, translated_vertices, triList, plys.vertices.size(), triList.size(), width, height);
    auto _g_end = glfwGetTime();

    cout << "hitGroupSize(CPU) : " << hitGroup.size() << endl;
    cout << "Time(ms) : " << (_end-_start)*1000 << endl;
    cout << "hitGroupSize(GPU) : " << hitGroup2.size() << endl;
    cout << "Time(ms) : " << (_g_end - _g_start) * 1000 << endl;

    parseRayResult("cpu_result.txt", cam.position, hitGroup, width, height);
    parseRayResult("gpu_result.txt", cam.position, hitGroup2, width, height);

    vector<AISTER_GRAPHICS_ENGINE::LineRenderer> lr(hitGroup.size());
    vector<AISTER_GRAPHICS_ENGINE::LineRenderer> lr2(hitGroup2.size());

    for (int i = 0; i < hitGroup.size(); i++) {
        lr[i].setShaderLine(cam.position, hitGroup[i].position, &lineShader);
    }

    for (int i = 0; i < hitGroup.size(); i++) {
        lr2[i].setShaderLine(cam.position, hitGroup2[i].position, &lineShader);
        lr2[i].setColor(glm::vec4(0, 0, 1, 1));
    }

    auto optixRay = getRayResult("test.txt");
    vector<AISTER_GRAPHICS_ENGINE::LineRenderer> lr3(optixRay.size());
    for (int i = 0; i < lr3.size(); i++) {
        lr3[i].setShaderLine(cam.position, optixRay[i], &lineShader);
        lr3[i].setColor(glm::vec4(0, 1, 0, 1));
    }
    
    cam.position = cam.position + glm::vec3(0,15,7);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    MVS_PCD testPCD;
    testPCD.readPrecomputed("example/tet.apq");

    cout << glm::to_string(testPCD.getTRS()) << endl;
    cout << testPCD.pts.size() << endl;
    cout << glm::to_string(testPCD.cams[0].getProjectionMatrix()) << endl;
    cout << glm::to_string(testPCD.cams[0].getViewMatrix()) << endl;
    cout << testPCD.faces.vIdx.size() << endl;
    cout << testPCD.faces.tIdx.size() << endl;
    cout << testPCD.faces.weight.size() << endl;

    AISTER_GRAPHICS_ENGINE::PCDRenderer pr;
    pr.setShader(&testPCD, &range_shader);
     
    auto vv = testPCD.getMinMaxWeight();

    AISTER_GRAPHICS_ENGINE::Camera cam2;
    cam2.position = testPCD.cams[3].position;
    //cam2.position = testPCD.get_3d_bbox().center + testPCD.get_3d_bbox().size;
    cam2._far = 1000.f;

    cam2.screenResolution = glm::vec2(width, height);
    cam2.direction = glm::normalize(testPCD.get_3d_bbox().center - cam2.position);


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // m2c values
    glm::mat4 m2c(glm::vec4(0.04992853, 0.001840242, -0.001938306, 0), glm::vec4(-4.371726E-05, -0.03569358, -0.03501383, 0), glm::vec4(0.002672364, -0.03496545, 0.03564093, 0), glm::vec4(1.21034, 0.8187663, -2.933632,1));
    
    // fx, fy, cx, cy, w, h, znear, zfar
    // Unity -> GL change to (fx, fy) -> (-fx, -fy)
    glm::mat4 persp = AISTER_GRAPHICS_ENGINE::getPerspectiveUsingK(572.4124, 573.5692, 320, 240, 640, 480, 0.01f, 4.5f);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glBlendEquation(GL_FUNC_ADD);
    glDepthFunc(GL_LESS);

    unsigned char* frameImage = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 4);
    unsigned char* frameImage2 = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 4);

    bool drawdepth = false;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDepthFunc(GL_LESS);
        renderer.Draw(cam, glm::vec4(1, 0, 0, 1), false);


        {
            glDepthFunc(GL_ALWAYS);
            //for(int i = 0; i<64;i++) boxRenderer[i].Draw(cam);
            //boxRenderer[0].Draw(cam);
            for (auto lrs : lr3)
                lrs.Draw(cam);

            /*for (auto lrs : lr2)
                lrs.Draw(cam);*/
        }

        glReadPixels(0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, frameImage2);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glDepthFunc(GL_LESS);
        //renderer.Draw(cam, glm::vec4(1, 0, 0, 1), false);
        pr.DrawTetra(cam2);

        glReadPixels(0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, frameImage);


        glfwGetWindowSize(window, &width, &height);
        cam.screenResolution = glm::vec2(width, height);
        rd.setCameraAllResol(glm::vec2(width, height));

        glfwSwapBuffers(window);
    }

    cv::Mat imgs(height, width, CV_8UC4);
    imgs.data = frameImage;
    cv::flip(imgs, imgs, 0);
    cv::imwrite("test.png", imgs);

    cv::Mat imgs2(height, width, CV_8UC4);
    imgs2.data = frameImage2;
    cv::flip(imgs2, imgs2, 0);
    cv::imwrite("test3.png", imgs2);

    glfwDestroyWindow(window);
    glfwTerminate();

}