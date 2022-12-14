#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"


#include<cmath>

using namespace glm;
using namespace std;
#include<vector>
// Function for intializing window

GLfloat lineVertices[10] ;

  unsigned int VBO1, VAO1, EBO1;  

void windowIntializeConfig(){
    
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

}
unsigned int loadCubemap(vector<std::string> faces)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrComponents;
    for (unsigned int i = 0; i < faces.size(); i++)
    {
        unsigned char *data = stbi_load(faces[i].c_str(), &width, &height, &nrComponents, 0);
        if (data)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        }
        else
        {
            std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
            stbi_image_free(data);
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}



vec3 temp1 = vec3(1.0f,0.0f,0.0f);
glm::mat4 view = glm::mat4(1.0f);
float x,y,z;
float theta1=0.0f;
int tx=0;
float phi1=0.0f;
const float radius = 3.0f;
float camX = sin(theta1) *radius ;
float camZ = -cos(theta1) *radius;
float camY = 0.0f;
float temp=0.0f;
float tempy=0.0f;
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
  
       if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
            tx=1;
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
            tx=0;
   
       
    
    
}

void framebuffer_size(GLFWwindow* window, int width, int height)
{

    glViewport(0, 0, width, height);
}

//Shaders

const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 COD;\n"
    "layout (location = 1) in vec2 uv;\n"
    "layout (location = 2) in vec3 normal;\n"
    "out vec3 COD1;\n"
     "out vec3 Normal;\n"
      "out vec3 WorldPos;\n"
      "out vec2 TexCoords;\n"
   
    "uniform mat4 view;\n"
    "uniform mat4 projection;\n"
    
    "void main()\n"
    "{\n"
    "TexCoords=uv;\n"
    "Normal=normal;\n"
   "WorldPos =COD;\n"
    "   gl_Position = projection*view*vec4(COD.x,COD.y,COD.z, 1.0);\n"
    "}\0";


const char *fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec2 TexCoords;\n"
"in vec3 WorldPos;\n"
"in vec3 Normal;\n"

"uniform vec3 albedo;\n"


"uniform vec3 camPos;\n"
"uniform int temp;\n"
"uniform samplerCube cubeMapTex; \n"
"const float PI = 3.14159265359;\n"
"const float metallic = 1.0f;\n"
"const float roughness =0.5f;\n"
"float chiGGX(float v)\n"
"{\n"
 "   return v > 0 ? 1 : 0;\n"
"}\n"
"float GGX_Distribution(vec3 n, vec3 h, float alpha)\n"
"{\n"
   " float NoH = dot(n,h);\n"
    "float alpha2 = alpha * alpha;\n"
    "float NoH2 = NoH * NoH;\n"
    "float den = NoH2 * alpha2 + (1 - NoH2);\n"
    "return (chiGGX(NoH) * alpha2) / ( PI * den * den );\n"
"}\n"
"float chiGGX1(float v)\n"
"{\n"
 "   return v > 0 ? 1 : 0;\n"
"}\n"
"float GGX_PartialGeometryTerm(vec3 v, vec3 n, vec3 h, float alpha)\n"
"{\n"
   " float VoH2 = max(dot(v,h),0.0);\n"
   " float chi = chiGGX1( VoH2 / max(dot(v,n),0.0) );\n"
   " VoH2 = VoH2 * VoH2;\n"
  "  float tan2 = ( 1 - VoH2 ) / VoH2;\n"
 "   return (chi * 2) / ( 1 + sqrt( 1 + alpha * alpha * tan2 ) );\n"
"}\n"

"vec3 fresnelSchlick(float cosTheta, vec3 F0)\n"
"{\n"
 "   return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);\n"
"}\n"


"float RadicalInverse_VdC(uint bits) \n"
"{\n"
 "    bits = (bits << 16u) | (bits >> 16u);\n"
   "  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);\n"
    " bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);\n"
     "bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);\n"
    " bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);\n"
   "  return float(bits) * 2.3283064365386963e-10; \n"
"}\n"

"vec2 Hammersley(uint i, uint N)\n"
"{\n"
"	return vec2(float(i)/float(N), RadicalInverse_VdC(i));\n"
"}\n"

"vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)\n"
"{\n"
"	float a = roughness*roughness;\n"

"	float phi = 2.0 * PI * Xi.x;\n"
"	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));\n"
"	float sinTheta = sqrt(1.0 - cosTheta*cosTheta);\n"
	
"	vec3 H;\n"
"	H.x = cos(phi) * sinTheta;\n"
"	H.y = sin(phi) * sinTheta;\n"
"	H.z = cosTheta;\n"
"	vec3 up          = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);\n"
"	vec3 tangent   = normalize(cross(up, N));\n"
"	vec3 bitangent = cross(N, tangent);\n"
	
"	vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;\n"
"	return normalize(sampleVec);\n"
"}\n"

"void main()\n"
"{		\n"
    "vec3 N = normalize(Normal);\n"
    "vec3 V = normalize(camPos - WorldPos);\n"  
   
  
"vec3 specular=vec3(0.0f);\n"
    "vec3 I= -V;\n"
    "vec3 R ;\n"
    
  "vec3 F0 = albedo;\n"
  "vec3 L ;\n"
"vec3 kD;\n"
"vec3 kS=vec3(0.0f);\n"
"const uint SAMPLE_COUNT = 64u;\n"
"for(uint i =0u; i<SAMPLE_COUNT;++i)\n"
"{\n"  
     "vec2 Xi = Hammersley(i, SAMPLE_COUNT);\n"
        "vec3 SampleVector = ImportanceSampleGGX(Xi, N, roughness);\n"
      
        "vec3 H = normalize(V + SampleVector);\n"
       "R = reflect(-H,N);\n"
        "float cosT=clamp(dot(SampleVector, N), 0.0, 1.0);\n"
        "float sinT = sqrt( 1 - cosT * cosT);\n"
       
        "float D = GGX_Distribution(N, H, roughness);   \n"
        "float G   =  GGX_PartialGeometryTerm(V,N,H,roughness)*GGX_PartialGeometryTerm(SampleVector,N,H,roughness); \n"   
        "vec3 F    = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);\n"

        "vec3 numerator    = texture(cubeMapTex,SampleVector).rgb*D* G * F *sinT; \n"
        "float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, H), 0.0)  + 0.05;\n" 
        "specular += numerator/denominator;\n"
        
        "kS += F;\n"
 "}\n"
    "kS = kS/SAMPLE_COUNT;\n"
    "specular = specular/SAMPLE_COUNT;\n"
     
  
    "vec3 color = kS*specular;\n" 
 
    "color = pow(color, vec3(1.0/2.2)); \n"
    "FragColor = vec4(color*5 , 1.0);\n"
"}\n\0";
const char *vertexShaderSource1 ="#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"

"out vec3 TexCoords;\n"

"uniform mat4 projection;\n"
"uniform mat4 view;\n"

"void main()\n"
"{\n"
    "vec3 Vert=aPos*10;\n"
    "TexCoords = Vert;\n"


   "vec4 pos = projection * view * vec4(Vert, 1.0);\n"
    "gl_Position = pos.xyww;\n"
"}  \n\0";

const char *fragmentShaderSource1 = "#version 330 core\n"
"out vec4 FragColor;\n"

"in vec3 TexCoords;\n"

"uniform samplerCube skybox;\n"

"void main()\n"
"{    \n"
    "FragColor = texture(skybox, TexCoords);\n"
"}\n\0";


int main()
{



    windowIntializeConfig();

    //Window creation
    GLFWwindow* window = glfwCreateWindow(800, 800, "Assignment-7", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size);


    if (glewInit() != GLEW_OK)
    { 
        std :: cout << "Failed to initialize GLEW" << std :: endl; 
        return -1;
     }



   
    // Compiling Shaders

    unsigned int vtxShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vtxShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vtxShader);
 
    int success;
    char log[512];
    glGetShaderiv(vtxShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vtxShader, 512, NULL, log);
        std::cout << "vertex shader compilation failed\n" << log << std::endl;
    }
 
    unsigned int fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragShader);
  
    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragShader, 512, NULL, log);
        std::cout << "frag shader compilation failed\n" << log << std::endl;
    }
    // Linking Shaders
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vtxShader);
    glAttachShader(shaderProgram, fragShader);
    glLinkProgram(shaderProgram);
  
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, log);
        std::cout << "Shader Link Failed\n" << log << std::endl;
    }
    glDeleteShader(vtxShader);
    glDeleteShader(fragShader);


    unsigned int vtxShader1 = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vtxShader1, 1, &vertexShaderSource1, NULL);
    glCompileShader(vtxShader1);
 
    int success1;
    char log1[512];
    glGetShaderiv(vtxShader1, GL_COMPILE_STATUS, &success1);
    if (!success1)
    {
        glGetShaderInfoLog(vtxShader1, 512, NULL, log1);
        std::cout << "vertex shader compilation failed\n" << log1<< std::endl;
    }
 
    unsigned int fragShader1 = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader1, 1, &fragmentShaderSource1, NULL);
    glCompileShader(fragShader1);
  
    glGetShaderiv(fragShader1, GL_COMPILE_STATUS, &success1);
    if (!success1)
    {
        glGetShaderInfoLog(fragShader1, 512, NULL, log1);
        std::cout << "frag shader compilation failed\n" << log1 << std::endl;
    }
    // Linking Shaders
    unsigned int shaderProgram1 = glCreateProgram();
    glAttachShader(shaderProgram1, vtxShader1);
    glAttachShader(shaderProgram1, fragShader1);
    glLinkProgram(shaderProgram1);
  
    glGetProgramiv(shaderProgram1, GL_LINK_STATUS, &success1);
    if (!success1) {
        glGetProgramInfoLog(shaderProgram1, 512, NULL, log1);
        std::cout << "Shader Link Failed\n" << log1 << std::endl;
    }
    glDeleteShader(vtxShader1);
    glDeleteShader(fragShader1);
    

    float skyboxVertices[] = {
        // positions          
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f
    };


 vector<std::string> faces;
    
        faces.push_back("./skybox/right.jpg");
        faces.push_back("./skybox/left.jpg");
        faces.push_back("./skybox/top.jpg");
        faces.push_back("./skybox/bottom.jpg");
        faces.push_back("./skybox/front.jpg");
        faces.push_back("./skybox/back.jpg");
    
    unsigned int cubemapTexture = loadCubemap(faces);


 

    vector<float> vertices1;
    vector<float>vertices2;
    vector<float>vertices3;
    vector<float>normals;
    vector<float>normals1;
    vector<float>vertices5;
    
   
    int stacks = 20;
    int slices = 20;
    const float pi = 3.14f;
    float theta1 = 0;
    float phi =0;
    vector<float> vertices;
    vector<float> Tvertices;
    vector<GLuint> indices;

    for (int i = 0; i <= stacks; ++i)
    {
        float V = (float)i / (float)stacks;
        float phi = V * pi;

        // loop through the slices.
        for (int j = 0; j <= slices; ++j)
        {
            float U = (float)j / (float)slices;
            float theta = U * (pi * 2);

            // use spherical coordinates to calculate the vertices.
            float x = cos(theta) * sin(phi);
            float y = cos(phi);
            float z = sin(theta) * sin(phi);
           
           
           
            vertices.push_back((x*0.3)+0.4);
            vertices.push_back(y*0.3);
            vertices.push_back(z*0.3);

            vertices1.push_back((x*0.3)-0.4);
            vertices1.push_back(y*0.3);
            vertices1.push_back(z*0.3);
          
            Tvertices.push_back(1-(theta/(2*pi)));
            Tvertices.push_back(phi/pi);
              normals.push_back(x+0.4);
            normals.push_back(y);
            normals.push_back(z);

            normals1.push_back(x-0.4);
            normals1.push_back(y);
            normals1.push_back(z);
        
            

        }
    }
    
    
    for (int i = 0; i < slices * stacks + slices; ++i){
        indices.push_back(GLuint(i));
        indices.push_back(GLuint(i + slices + 1));
        indices.push_back(GLuint(i + slices));

        indices.push_back(GLuint(i + slices + 1));
        indices.push_back(GLuint(i));
        indices.push_back(GLuint(i + 1));
    }

    
    
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER,sizeof(GLfloat)*vertices.size(), vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*indices.size(), indices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
    unsigned int VBO_texture; 
    glGenBuffers(1, &VBO_texture); 
     glBindBuffer(GL_ARRAY_BUFFER, VBO_texture); 
    glBufferData(GL_ARRAY_BUFFER, 9*Tvertices.size(), Tvertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0); 
    glEnableVertexAttribArray(1);
    unsigned int VBO_normal; 
    glGenBuffers(1, &VBO_normal); 
     glBindBuffer(GL_ARRAY_BUFFER, VBO_normal); 
    glBufferData(GL_ARRAY_BUFFER, 9*normals.size(), normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),  (void*)(0 * sizeof(float))); 
    glEnableVertexAttribArray(2);
    glEnable(GL_DEPTH_TEST); 
    
    
    unsigned int VBO1, VAO1, EBO1;
    glGenVertexArrays(1, &VAO1);
    glGenBuffers(1, &VBO1);
    glGenBuffers(1, &EBO1);
    glBindVertexArray(VAO1);
    glBindBuffer(GL_ARRAY_BUFFER, VBO1);
    glBufferData(GL_ARRAY_BUFFER,sizeof(GLfloat)*vertices1.size(), vertices1.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO1);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*indices.size(), indices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
    unsigned int VBO_texture1; 
    glGenBuffers(1, &VBO_texture1); 
     glBindBuffer(GL_ARRAY_BUFFER, VBO_texture1); 
    glBufferData(GL_ARRAY_BUFFER, 9*Tvertices.size(), Tvertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0); 
    glEnableVertexAttribArray(1);
    unsigned int VBO_normal1; 
    glGenBuffers(1, &VBO_normal1); 
     glBindBuffer(GL_ARRAY_BUFFER, VBO_normal1); 
    glBufferData(GL_ARRAY_BUFFER, 9*normals1.size(), normals1.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),  (void*)(0 * sizeof(float))); 
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glBindVertexArray(0); 

    

  unsigned int skyboxVAO, skyboxVBO;
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, 9*sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);


    while (!glfwWindowShouldClose(window))
    {

        processInput(window);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
       glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
        glUseProgram(shaderProgram);
       glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
        
     
int temp=0;

unsigned int tempLoc= glGetUniformLocation(shaderProgram, "temp");
       view = glm::lookAt(glm::vec3(camX, camY, camZ), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0));
        glm::mat4 projectionMatrix = mat4(1.0f);
         glm :: vec3 camPos = vec3(camX, camY, camZ);
        projectionMatrix =glm::perspective(glm::radians(45.0f),1.0f,0.1f,100.0f);
        
        glm :: vec3 albedo = vec3(0.447f, 0.157f, 0.118f);
        unsigned int viewLoc= glGetUniformLocation(shaderProgram, "view");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
         unsigned int viewPosLoc= glGetUniformLocation(shaderProgram, "camPos");
        glUniform3fv(viewPosLoc, 1, glm::value_ptr(camPos));
        unsigned int albedoLoc= glGetUniformLocation(shaderProgram, "albedo");
        glUniform3fv(albedoLoc, 1, glm::value_ptr(albedo));
       unsigned int projLoc= glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
        glBindVertexArray(VAO); 
        glDrawElements(GL_TRIANGLES, 9*vertices.size() , GL_UNSIGNED_INT, 0);
       
        glUseProgram(shaderProgram);
       glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
        
        albedo = vec3(01.0f, 0.86f, 0.57f);
       glUniform3fv(albedoLoc, 1, glm::value_ptr(albedo));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        temp=1;
        glUniform3fv(viewPosLoc, 1, glm::value_ptr(camPos));
       glUniform1i(tempLoc,temp);
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
        glBindVertexArray(VAO1); 
    
        
        glDrawElements(GL_TRIANGLES, 9*vertices1.size() , GL_UNSIGNED_INT, 0);
        tx==1?glPolygonMode(GL_FRONT_AND_BACK, GL_POINT):glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
        glPointSize(20);
        
       
        glDepthFunc(GL_LEQUAL);  
        glUseProgram(shaderProgram1);

        unsigned int viewLoc1= glGetUniformLocation(shaderProgram1, "view");
        glUniformMatrix4fv(viewLoc1, 1, GL_FALSE, glm::value_ptr(view));
     
        unsigned int projLoc1= glGetUniformLocation(shaderProgram1, "projection");
        glUniformMatrix4fv(projLoc1, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
       
        glBindVertexArray(skyboxVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        glDepthFunc(GL_LESS); 
    


        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);
    glDeleteProgram(shaderProgram1);


    glfwTerminate();
    return 0;
}


