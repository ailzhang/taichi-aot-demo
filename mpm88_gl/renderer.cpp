#include "renderer.h"

namespace demo {
namespace {
bool CreateAndCompileShader(GLuint &shader_id, GLenum shader_type, const char* src) {
    const GLchar *source = (GLchar*)src;
    if (!source || !std::strlen(source))
    {
        return false;
    }

    shader_id = glCreateShader(shader_type);

    if (shader_id == 0)
    {
        LogError("CreateShader() failed!\n");
        return false;
    }

    glShaderSource(shader_id, 1, &source, NULL);
    glCompileShader(shader_id);

    GLint status;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &status);

    if (status == 0)
    {
        GLint logLength;
        glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &logLength);

        if (logLength > 0)
        {
            GLchar *log = (GLchar *)malloc(logLength);
            glGetShaderInfoLog(shader_id, logLength, &logLength, log);
            printf("CreateAndCompileShader compile log:\n %s\n", log);
            free(log);
        }

        glDeleteShader(shader_id);
        return false;
    }

    return true;       
}
}

// Shader
Shader::Shader()
    : shader_id(0)
    , shader_type(GL_VERTEX_SHADER)
{
}

Shader::~Shader()
{
    if (shader_id && glIsShader(shader_id))
    {
        glDeleteShader(shader_id);
        shader_id = 0;
    }
}

bool Shader::InitFromSource(const std::string& src, GLenum shader_type)
{
    if (!CreateAndCompileShader(shader_id, shader_type, src.c_str()))
    {
        LogError("CreateAndCompileShader(): failed: \n %s\n", src.c_str());
        return false;
    }
    this->shader_type = shader_type;

    return true;
}

std::unique_ptr<Shader> Shader::CreateFromSource(
    const std::string& src, GLenum shader_type)
{
    if (src.empty())
    {
        return nullptr;
    }

    auto shader = std::make_unique<Shader>();

    if (nullptr != shader)
    {
        if (!shader->InitFromSource(src, shader_type))
        {
            return nullptr;
        }
    }

    return shader;
}

// Program
Program::Program()
    : program_id(0)
    , vert_shader(nullptr)
    , frag_shader(nullptr)
    , comp_shader(nullptr)
{
}

Program::~Program()
{
    if (program_id && glIsProgram(program_id))
    {
        glDeleteProgram(program_id);
        program_id = 0;
    }
}

bool Program::AttachShader(Shader* shader)
{
    if (0 == program_id)
    {
        LogError("Program::AttachShader(): program_id == 0!\n");
        return false;
    }

    if (nullptr != shader)
    {
        if (0 == shader->shader_id)
        {
            LogError("Program::AttachShader(): shader_id == 0!\n");
            return false;
        }
        glAttachShader(program_id, shader->shader_id);
    }

    return true;
}
bool Program::Link()
{
    if (0 == program_id)
    {
        LogError("Program::Link(): program_id == 0!\n");
        return false;
    }

    if(!AttachShader(vert_shader.get())){
        LogError("Program::Link(): AttachShader(vert_shader) failed!\n");
        return false;
    }

    if (!AttachShader(frag_shader.get())) {
        LogError("Program::Link(): AttachShader(frag_shader) failed!\n");
    }

    if (!AttachShader(comp_shader.get())) {
        LogError("Program::Link(): AttachShader(comp_shader) failed!\n");
    }

   
    glLinkProgram(program_id);

    GLint status;
    glGetProgramiv(program_id, GL_LINK_STATUS, &status);

    if(status == 0)
    {
        GLint logLength;
        glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &logLength);
        if (logLength > 1)
        {
            GLchar *log = (GLchar *)malloc(logLength);
            glGetProgramInfoLog(program_id, logLength, &logLength, log);
            LogError("Program::Link: Link program error log: %s\n", log);
            free(log);
        }
    }

    return true;
}

bool Program::Enable()
{
    if (0 == program_id)
    {
        LogError("Program: program_id == 0!\n");
        return false;
    }

    glUseProgram(program_id);

    return true;
}
bool Program::Disable()
{
    glUseProgram(0);

    return true;
}

bool Program::Init()
{
    if (0 != program_id)
    {
        LogError("Program: program_id has been initialized!\n");
        return false;
    }

    program_id = glCreateProgram();

    if (0 == program_id)
    {
        return false;
    }

    return true;
}

std::unique_ptr<Program> Program::Create()
{
    auto program = std::make_unique<Program>();

    if (nullptr != program)
    {
        if (!program->Init())
        {
            return nullptr;
        }
    }

    return program;
}

// Renderer
const std::string Renderer::VERT_SRC = R"(
    #version 320 es

    precision highp float;

    layout(location = 0) in vec2 viPos;

    void main() {

        gl_Position = vec4(viPos.xy*2.0 - 1.0, 0, 1);

        gl_PointSize = 4.0f;
     }
    )";

const std::string Renderer::FRAG_SRC = R"(
    #version 320 es

    precision highp float;

    layout(location = 0) out vec4 fragColor;

    void main() {
        fragColor = vec4(0.8, 0.4, 0.1, 1.0);
    }
    )";

Renderer::Renderer(GLuint pos, GLsizei count): pos_(pos), count_(count) {
  program_ = Program::Create();
  
  auto vert_shader = Shader::CreateFromSource(VERT_SRC, GL_VERTEX_SHADER);
  if (vert_shader == nullptr) {
    LogError("Init vert_shader failed\n");
    return;
  }

  auto frag_shader = Shader::CreateFromSource(FRAG_SRC, GL_FRAGMENT_SHADER);
  if (frag_shader == nullptr) {
    LogError("Init frag_shader failed\n");
    return;
  }

  program_->frag_shader = std::move(frag_shader);
  program_->vert_shader = std::move(vert_shader);

  if (!program_->Link()) {
    LogError("program link failed");
    return;
  }

  glGenVertexArrays(1, &vertex_array_buffer_); PrintGLError();  
  glBindVertexArray(vertex_array_buffer_); PrintGLError();
  {
      glBindBuffer(GL_ARRAY_BUFFER, pos_); PrintGLError();  
      GLint posAttrib = glGetAttribLocation(program_->program_id, "viPos"); PrintGLError();  
      glEnableVertexAttribArray(posAttrib); PrintGLError();  
      glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL); PrintGLError();
  }
  glBindVertexArray(0); PrintGLError();  // unbind 
}

Renderer::~Renderer() {
  if (vertex_array_buffer_ != 0) {
    glDeleteVertexArrays(1, &vertex_array_buffer_);
    vertex_array_buffer_ = 0;
  }
}

bool Renderer::Render() {
    program_->Enable();
    {
        glEnable(GL_PROGRAM_POINT_SIZE);
        glBindVertexArray(vertex_array_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, pos_);
        glDrawArrays(GL_POINTS, 0, count_);
        glBindVertexArray(0);
    }
    program_->Disable();

    return true;
}
}