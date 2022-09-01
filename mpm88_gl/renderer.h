#pragma once

#include <glad/gl.h>
#include "gl_utils.h"

#include <vector>
#include <string>
#include <cstring>

namespace demo {

class Shader
{
    UNCOPYABLE(Shader);
public:
    Shader();

    virtual ~Shader();
public:
    /*
     * Create gl shader from source
     * @src : the source code of shader
     * @type: GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPUTE_SHAER
    */
    static std::unique_ptr<Shader> CreateFromSource(
        const std::string& src, GLenum shader_type);
public:
    bool InitFromSource(const std::string& src, GLenum shader_type);
public:
    GLuint shader_id;
    GLenum shader_type;
};

class Program
{
    UNCOPYABLE(Program);
public:
    Program();

    virtual ~Program();
public:
    bool Link();

    bool Enable();

    bool Disable();
public:
    /*
     * Create gl program
    */
    static std::unique_ptr<Program> Create();

protected:
    bool Init();
    bool AttachShader(Shader* shader);
public:
    GLuint program_id;
    std::unique_ptr<Shader> vert_shader;
    std::unique_ptr<Shader> frag_shader;
    std::unique_ptr<Shader> comp_shader;
};


class Renderer {
public:

Renderer(GLuint pos, GLsizei count);
~Renderer();

bool Render();

private:
GLuint vertex_array_buffer_{0};
GLuint root_buffer_{0};  // TODO: remove
GLuint pos_;
GLsizei count_;
std::unique_ptr<Program> program_;


static const std::string FRAG_SRC;
static const std::string VERT_SRC;
};
}