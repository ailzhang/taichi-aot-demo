#pragma once
#include <glad/gl.h>
#include <memory>
#include <string>

#define UNCOPYABLE(ctype)              \
    public:                            \
        ctype(const ctype &) = delete; \
        ctype& operator = (const ctype &) = delete

#define LogInfo    printf
#define LogWarning printf
#define LogError   printf

#define PrintGLError() {\
    switch (glGetError()) \
    {\
    case GL_NO_ERROR:\
        break;\
    case GL_INVALID_ENUM:\
        printf("FILE: %s, LINE: %d, GL_INVALID_ENUM\n", __FILE__, __LINE__);\
        break;\
    case GL_INVALID_VALUE:\
        printf("FILE: %s, LINE: %d, GL_INVALID_VALUE\n", __FILE__, __LINE__);\
        break;\
    case GL_INVALID_OPERATION:\
        printf("FILE: %s, LINE: %d, GL_INVALID_OPERATION\n", __FILE__, __LINE__);\
        break;\
    case GL_INVALID_FRAMEBUFFER_OPERATION:\
        printf("FILE: %s, LINE: %d, GL_INVALID_FRAMEBUFFER_OPERATION\n", __FILE__, __LINE__);\
        break;\
    case GL_OUT_OF_MEMORY:\
        printf("FILE: %s, LINE: %d, GL_OUT_OF_MEMORY\n", __FILE__, __LINE__);\
        break;\
    default:\
        printf("FILE: %s, LINE: %d, Other Error\n", __FILE__, __LINE__);\
        break;\
    }\
}