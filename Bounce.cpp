#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>

#include <cmath>
#include <cstdlib>
#include <ctime>

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>


const char* vertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec2 aPos;

    uniform float time;

    void main() {
        gl_Position = vec4(aPos.xy, 0.0, 1.0);
    }
)";

const char* geometryShaderSource = R"(
    #version 430 core
    layout (points) in;
    layout (triangle_strip, max_vertices = 168) out;

    out float border;

    void main() {
        float del = 0.5f;
        float rad = 0.005;
        float borderThickness = 0.0025;
        for (float theta = 0.0; theta <= 2.0 * 3.14159265359; theta += del) {
            // For the normal thang
            vec3 position;
            position.x = gl_in[0].gl_Position.x;
            position.y = gl_in[0].gl_Position.y;
            position.z = gl_in[0].gl_Position.z;
            gl_Position = vec4(position, 1.0);
            border = 1.0;
            EmitVertex();

            position.x = gl_in[0].gl_Position.x + rad * cos(theta);
            position.y = gl_in[0].gl_Position.y + rad * sin(theta);
            position.z = gl_in[0].gl_Position.z;
            gl_Position = vec4(position, 1.0);
            border = 1.0;
            EmitVertex();

            position.x = gl_in[0].gl_Position.x + rad * cos(theta + del);
            position.y = gl_in[0].gl_Position.y + rad * sin(theta + del);
            position.z = gl_in[0].gl_Position.z;
            gl_Position = vec4(position, 1.0);
            border = 1.0;
            EmitVertex();

            position.x = gl_in[0].gl_Position.x + (rad + borderThickness)  * cos(theta);
            position.y = gl_in[0].gl_Position.y + (rad + borderThickness) * sin(theta);
            position.z = gl_in[0].gl_Position.z;
            gl_Position = vec4(position, 1.0);
            border = 0.0;
            EmitVertex();

            position.x = gl_in[0].gl_Position.x + (rad + borderThickness)  * cos(theta + del);
            position.y = gl_in[0].gl_Position.y + (rad + borderThickness) * sin(theta + del);
            position.z = gl_in[0].gl_Position.z;
            gl_Position = vec4(position, 1.0);
            border = 0.0;
            EmitVertex();

            EndPrimitive();
        }
    }
)";

const char* fragmentShaderSource = R"(
    #version 430 core
    in float border;
    out vec4 FragColor;

    uniform float time;

    vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }

    void main() {
        vec3 hsv = vec3(mod(time/10, 1.0), 1.0, 1.0);
        vec3 borderColor = vec3(0.0, 0.0, 0.0);

        vec3 rgb =  mix(hsv2rgb(hsv), borderColor, 1-border);
        // vec3 rgb =  mix(vec3(1.0, 0.0, 0.0), borderColor, 1-border);
        FragColor = vec4(rgb , 1.0);
    }
)";

const char* computeShaderSource = R"(
    #version 430

    layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

    uniform float deltaTime;
    uniform float numVerts;

    // Buffer to read and write data
    layout(std430, binding = 0) buffer InPosBuffer {
        vec2 inPositions[];
    };

    layout(std430, binding = 1) buffer OutPosBuffer {
        vec2 outPositions[];
    };

    layout(std430, binding = 2) buffer InVelBuffer {
        vec2 inVelocities[];
    };

    layout(std430, binding = 3) buffer OutVelBuffer {
        vec2 outVelocities[];
    };

    float speed = 0.1f;
    // vec2 acc = vec2(0, -10);
    vec2 acc = vec2(0, 0);

    float dampY = 1.0f;
    float dampX = 1.0f;
    float gravityCoeff = 0.0001;

    float rad = 0.005;
    float borderThickness = 0.0025;

    void main() {

        uint globalIndex = gl_GlobalInvocationID.x;

        vec2 pos = inPositions[globalIndex];
        vec2 vel = inVelocities[globalIndex];


        float effectiveNegBorder = -1 + rad;
        if(pos.y < effectiveNegBorder && vel.y < 0){
            vel.y = -1 * vel.y * dampY;
            // pos.y = 2 * effectiveNegBorder - pos.y;
        } else if(pos.y > -effectiveNegBorder && vel.y > 0){
            vel.y = -1 * vel.y * dampY;
            // pos.y = -2 * effectiveNegBorder - pos.y;
        }

        if(pos.x < effectiveNegBorder && vel.x < 0){
            vel.x = -1 * vel.x * dampX;
            // pos.x = 2 * effectiveNegBorder - pos.x;
        } else if(pos.x > -effectiveNegBorder && vel.x > 0){
            vel.x = -1 * vel.x * dampX;
            // pos.x = -2*effectiveNegBorder - pos.x;
        }

        for(int i = 0; i < numVerts; i++){
            if(i == globalIndex)
                continue;
            
            float effRad = rad + borderThickness;
            float dis = distance(pos, inPositions[i]);
            float delDis = abs(2*effRad - dis)/2.0;
            vec2 outNormal = normalize(inPositions[i] - pos);

            acc += outNormal * gravityCoeff / pow(dis, 2);

            if(dis < 2*effRad){
                pos = pos - outNormal * delDis;

                float num = dot(vel - inVelocities[i], pos - inPositions[i]);
                num /= pow(distance(pos, inPositions[i]), 2);
                vel = vel - num * (pos - inPositions[i]);
            }
        }

        vel = vel + acc * deltaTime;

        outPositions[globalIndex] = pos + vel * deltaTime;
        outVelocities[globalIndex] = vel;
    }
)";

const int num_Vertices = 9;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

void setInputAndOutputBuffersForCompute(
        GLuint *inputPosBuffer, GLuint *outputPosBuffer,
        GLuint *inputVelBuffer, GLuint *outputVelBuffer,
        glm::vec2 *inputPos, glm::vec2 *inputVel
    ){
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, *inputPosBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, num_Vertices * sizeof(glm::vec2), inputPos, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, *inputPosBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, *outputPosBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, num_Vertices * sizeof(glm::vec2), nullptr, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, *outputPosBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, *inputVelBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, num_Vertices * sizeof(glm::vec2), inputVel, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, *inputVelBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, *outputVelBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, num_Vertices * sizeof(glm::vec2), nullptr, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, *outputVelBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

std::pair<glm::vec2*, glm::vec2*> useAndDispatchComputeShader(
        GLuint *computeProgram, 
        GLuint *outPosBuffer, GLuint *outVelBuffer,
        float time, GLint timeLoc, int divisor
    ){
    // Use the compute shader program
    glUseProgram(*computeProgram);
    glUniform1f(timeLoc, time);

    // Dispatch the compute shader
    glDispatchCompute((num_Vertices + divisor - 1)/divisor, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Read the result from the output buffer
    glm::vec2 *outputPos = new glm::vec2[num_Vertices];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, *outPosBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, num_Vertices * sizeof(glm::vec2), outputPos);

    glm::vec2 *outputVel = new glm::vec2[num_Vertices];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, *outVelBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, num_Vertices * sizeof(glm::vec2), outputVel);

    return std::pair<glm::vec2*, glm::vec2*>(outputPos, outputVel);
}

int createAndCompileComputeShaderandComputeProgram(GLuint *computeShader, GLuint *computeProgram){
    // Load and compile the compute shader
    *computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(*computeShader, 1, &computeShaderSource, nullptr);
    glCompileShader(*computeShader);

    // Check for shader compilation errors
    GLint success;
    glGetShaderiv(*computeShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(*computeShader, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "Compute shader compilation failed:\n" << infoLog << std::endl;
        return -1;
    }

    // Create a compute shader program
    *computeProgram = glCreateProgram();
    glAttachShader(*computeProgram, *computeShader);
    glLinkProgram(*computeProgram);

    // Check for program linking errors
    glGetProgramiv(*computeProgram, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetProgramInfoLog(*computeProgram, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "Program linking failed:\n" << infoLog << std::endl;
        return -1;
    }

    return 0;
}

int createAndCompileVertexFragmentShaderAndProgram(GLuint *vertexShader, GLuint *geomteryShader, GLuint *fragmentShader, GLuint *shaderProgram){
    *vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(*vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(*vertexShader);

    GLint success;
    glGetShaderiv(*vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(*vertexShader, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "Vertex shader compilation failed:\n" << infoLog << std::endl;
        return -1;
    }

    *geomteryShader = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(*geomteryShader, 1, &geometryShaderSource, nullptr);
    glCompileShader(*geomteryShader);

    glGetShaderiv(*geomteryShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(*geomteryShader, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "Geometry shader compilation failed:\n" << infoLog << std::endl;
        return -1;
    }

    *fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(*fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(*fragmentShader);

    glGetShaderiv(*fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(*fragmentShader, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "Frag shader compilation failed:\n" << infoLog << std::endl;
        return -1;
    }

    *shaderProgram = glCreateProgram();
    glAttachShader(*shaderProgram, *vertexShader);
    glAttachShader(*shaderProgram, *geomteryShader);
    glAttachShader(*shaderProgram, *fragmentShader);
    glLinkProgram(*shaderProgram);

    glGetProgramiv(*shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetProgramInfoLog(*shaderProgram, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "Shader Program linking failed:\n" << infoLog << std::endl;
        return -1;
    }

    return 0;
}

void setVAOandVBOforRender(GLuint *VAO, GLuint *VBO, glm::vec2 *VBOData){

    glBindVertexArray(*VAO);
    glBindBuffer(GL_ARRAY_BUFFER, *VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * (num_Vertices), VBOData, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)0);
    glEnableVertexAttribArray(0);
}

void updateDeltaTime() {
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

float updateFPS(GLFWwindow* window){
    float FPS = 1.0/deltaTime;
    std::string windowTitle = "Bounce Bounce | FPS: " + std::to_string(FPS);
    glfwSetWindowTitle(window, windowTitle.c_str());

    // std::cout << windowTitle << std::endl;

    return FPS;
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Set OpenGL version to 4.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    glfwWindowHint(GLFW_SAMPLES, 4); // anti-aliasing

    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(800, 600, "Bounce Bounce", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Set clear color
    glClearColor(0.529f, 0.808f, 0.922f, 1.0f);

    // To make it RGB-A
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Anti-aliasing
    glEnable(GL_MULTISAMPLE);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    // glfwMaximizeWindow(window);

    // Check if the required OpenGL version is supported
    if (!GLEW_VERSION_4_3) {
        std::cerr << "OpenGL 4.3 is not supported" << std::endl;
        return -1;
    }

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    glm::vec2 *initialPositions = new glm::vec2[num_Vertices];
    glm::vec2 *velocities = new glm::vec2[num_Vertices];
    float heightWidthFactor = 0.9f;
    float sqSide = 1.5f;
    for (int i = 0; i < num_Vertices; ++i) {
        float randomX = -sqSide/2 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / sqSide));
        float randomY = -sqSide/2 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / sqSide));

        float randomX1 = -1 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0));
        float randomY1 = -1 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0));   

        // initialPositions[i] = glm::vec2((float)i*heightWidthFactor/(num_Vertices), (float)i*heightWidthFactor/(num_Vertices));
        // initialPositions[i] = glm::vec2(randomX, randomY);
        // velocities[i] = glm::vec2(randomX1/10, randomY1/10);
        velocities[i] = glm::vec2(0.0f,  0.0f);
    }

    float num_on_side = sqrt(num_Vertices);
    float spacing = sqSide/num_on_side;

    for(int i = 0; i < num_on_side; i++){
        for(int j = 0; j < num_on_side; j++){
                float x = -sqSide/2 + i * spacing;
                float y = -sqSide/2 + j * spacing;
                initialPositions[(int)num_on_side * i + j] = glm::vec2(x, y);
        }
    }

    // for(int i = 0; i < num_Vertices; i++)
    //     std::cout << glm::to_string(initialPositions[i]) << std::endl;


    // ----------------------------------------------------- COLOR SHADER PART --------------------------------------------------
    GLuint vertexShader, geometryShader, fragmentShader, shaderProgram;
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    
    createAndCompileVertexFragmentShaderAndProgram(&vertexShader, &geometryShader, &fragmentShader, &shaderProgram);
    setVAOandVBOforRender(&VAO, &VBO, initialPositions);

    glUseProgram(shaderProgram);
    GLint timeShaderLoc = glGetUniformLocation(shaderProgram, "time");
    glUseProgram(0);


    // ----------------------------------------------------- COMPUTE SHADER PART ------------------------------------------------
    // Create buffers
    GLuint inputPosBuffer,  outputPosBuffer, inputVelBuffer, outputVelBuffer;
    GLuint computeShader, computeProgram;
    glGenBuffers(1, &inputPosBuffer);
    glGenBuffers(1, &outputPosBuffer);
    glGenBuffers(1, &inputVelBuffer);
    glGenBuffers(1, &outputVelBuffer);
    
    createAndCompileComputeShaderandComputeProgram(&computeShader, &computeProgram); 
    // For the first time, inputData is init value after that its always outputData 
    setInputAndOutputBuffersForCompute(&inputPosBuffer, &outputPosBuffer, &inputVelBuffer, &outputVelBuffer, initialPositions, velocities);       

    glUseProgram(computeProgram);
    GLint timeComputeLoc = glGetUniformLocation(computeProgram, "deltaTime");
    GLint numVertsComputeLoc = glGetUniformLocation(computeProgram, "numVerts");
    glUniform1f(numVertsComputeLoc, num_Vertices);
    glUseProgram(0);

    glm::vec2 *outPositions, *outVelocities;
    std::pair<glm::vec2*, glm::vec2*>  outputs;

    float sumFPS = 0;
    float numFrames = 0;

    while(!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        updateDeltaTime();

        sumFPS += updateFPS(window);
        numFrames++;
        // ------------------------------------------------- Running Compute Shader ----------------------------------------
        outputs = useAndDispatchComputeShader(&computeProgram, &outputPosBuffer, &outputVelBuffer, deltaTime, timeComputeLoc, 32);
        outPositions = outputs.first;
        outVelocities = outputs.second;
        setInputAndOutputBuffersForCompute(&inputPosBuffer, &outputPosBuffer, &inputVelBuffer, &outputVelBuffer, outPositions, outVelocities);       

        // std::cout << deltaTime << ": " << glm::to_string(outPositions[0]) << " " << glm::to_string(outVelocities[0]) << std::endl;
        // std::cout << "Velocity of 0: " << glm::to_string(outVelocities[0]) << std::endl;

        glFinish();
        // ------------------------------------------------ Running Color  Shader ----------------------------------------
        glUseProgram(shaderProgram);
        glUniform1f(timeShaderLoc, static_cast<float>(glfwGetTime()));

        setVAOandVBOforRender(&VAO, &VBO, outPositions);

        glDrawArrays(GL_POINTS, 0, num_Vertices);

        glfwSwapBuffers(window);
        glfwPollEvents();

    }

    float avgFPS = sumFPS/numFrames;
    std::cout << "Average FPS: " << avgFPS << std::endl;

    // Clean up
    delete[] outPositions;
    glDeleteProgram(computeProgram);
    glDeleteProgram(shaderProgram);

    glDeleteShader(computeShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glDeleteBuffers(1, &inputPosBuffer);
    glDeleteBuffers(1, &outputPosBuffer);
    glDeleteBuffers(1, &inputVelBuffer);
    glDeleteBuffers(1, &inputPosBuffer);

    // Terminate GLFW
    glfwTerminate();

    return 0;
}
