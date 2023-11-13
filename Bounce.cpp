#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

// Size of the data array
const int num_Vertices = 20;

float deltaTime = 0.0f;
float lastFrame = 0.0f;


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
        float rad = 0.02;
        float borderThickness = 0.01;
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
        FragColor = vec4(rgb , 1.0);
    }
)";

const char* computeShaderSource = R"(
    #version 430

    layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

    uniform float deltaTime;

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
    vec2 acc = vec2(0, -10);

    float dampY = 0.75f;
    float stoppingY = 0.001;
    float dampX = 0.75f;
    float stoppingX = 0.05;

    void main() {
        float rad = 0.02;

        uint globalIndex = gl_GlobalInvocationID.x;

        vec2 pos = inPositions[globalIndex];
        vec2 vel = inVelocities[globalIndex];

        vel = vel + acc * deltaTime;

        float effectiveNegBorder = -1 + rad;
        if(pos.y < effectiveNegBorder){
            float newY = -1 * vel.y * dampY;
            vel.y = abs(newY) > stoppingY ? newY : 0.0f;
            pos.y = 2 * effectiveNegBorder - pos.y;
        }

        if(pos.x < effectiveNegBorder){
            float newX = -1 * vel.x * dampX;
            vel.x = abs(newX) > stoppingX ? newX : 0.0f;
            pos.x = 2 * effectiveNegBorder - pos.x;
        } else if(pos.x > -effectiveNegBorder){
            float newX = -1 * vel.x * dampX;
            vel.x = abs(newX) > stoppingX ? newX : 0.0f;
            pos.x = -2*effectiveNegBorder - pos.x;
        }

        outPositions[globalIndex] = pos + vel * deltaTime;
        outVelocities[globalIndex] = vel;
    }
)";

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

    glm::vec2 *initialPositions = new glm::vec2[num_Vertices];
    glm::vec2 *velocities = new glm::vec2[num_Vertices];
    float heightWidthFactor = 0.66f;
    for (int i = 0; i < num_Vertices; ++i) {
        initialPositions[i] = glm::vec2((float)i*heightWidthFactor/(num_Vertices), (float)i*heightWidthFactor/(num_Vertices));
        velocities[i] = glm::vec2(0.5f, 0.0f);
    }


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
    glUseProgram(0);

    glm::vec2 *outPositions, *outVelocities;
    std::pair<glm::vec2*, glm::vec2*>  outputs;

    while(!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        // ------------------------------------------------- Running Compute Shader ----------------------------------------
        updateDeltaTime();
        outputs = useAndDispatchComputeShader(&computeProgram, &outputPosBuffer, &outputVelBuffer, deltaTime, timeComputeLoc, 32);
        outPositions = outputs.first;
        outVelocities = outputs.second;
        setInputAndOutputBuffersForCompute(&inputPosBuffer, &outputPosBuffer, &inputVelBuffer, &outputVelBuffer, outPositions, outVelocities);       

        // std::cout << deltaTime << ": " << glm::to_string(outPositions[0]) << " " << glm::to_string(outVelocities[0]) << std::endl;
        // std::cout << "Velocity of 0: " << glm::to_string(outVelocities[0]) << std::endl;

        // ------------------------------------------------ Running Color  Shader ----------------------------------------
        glUseProgram(shaderProgram);
        glUniform1f(timeShaderLoc, static_cast<float>(glfwGetTime()));

        setVAOandVBOforRender(&VAO, &VBO, outPositions);

        glDrawArrays(GL_POINTS, 0, num_Vertices);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

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
