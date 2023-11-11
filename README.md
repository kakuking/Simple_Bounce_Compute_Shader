# A simple OpenGL Bouncing "Balls" Program
Created using GLEW, GLFW3, and GLM

The code is a simple program that implements the following:
There are 10 vertices, that initially spawn from (0.0, 0.0) --> (0.9, 0.9).

Using a compute shader, these are moved around and then are rendered using a simple vertex and frag shader.

To run: 
> g++ Bounce.cpp -o Bounce -lglfw3 -lopengl32 -lglew32 -lglu32 -lgdi32 -static-libstdc++

> .\Bounce.exe