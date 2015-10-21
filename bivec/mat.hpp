#pragma once
#include <vector>

typedef std::vector<float> vec;
typedef std::vector<vec> mat;

/*
namespace Vector {
    inline vec Zero(int size) {
        return vec(size, 0);
    }

    inline vec Random(int size) {
        vec v;
        for (int i = 0; i < size; ++i) {
            v.push_back(rand());
        }
        return v;
    }
}

namespace Matrix {
    inline mat Zero(int rows, int cols) {
        return mat(rows, vec(cols));
    }

    inline mat Random(int rows, int cols) {
        mat m;
        for (int i = 0; i < rows; ++i) {
            m.push_back(Vector::Random(cols));
        }
        return m;
    }
}
*/

static vec& operator+=(vec& lhs, const vec& rhs) {
    for (size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] += rhs[i];
    }
    return lhs;
}

inline vec& operator-=(vec& lhs, const vec& rhs) {
    for (size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] += rhs[i];
    }
    return lhs;
}

inline vec& operator*=(vec& lhs, float rhs) {
    for (size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] *= rhs;
    }
    return lhs;
}

inline vec& operator/=(vec& lhs, float rhs) {
    return lhs *= 1 / rhs;
}

inline vec operator-(vec rhs) {
    for (size_t i = 0; i < rhs.size(); ++i) {
        rhs[i] = -rhs[i]; // TODO optimize
    }
    return rhs;
}

inline vec operator+(vec lhs, const vec& rhs) {
    lhs += rhs;
    return lhs;
}

inline vec operator-(vec lhs, const vec& rhs) {
    lhs -= rhs;
    return lhs;
}

inline vec operator/(vec lhs, float rhs) {
    lhs /= rhs;
    return lhs;
}

inline vec operator*(vec lhs, float rhs) {
    lhs *= rhs;
    return lhs;
}

inline vec operator*(float lhs, vec rhs) {
    rhs *= lhs;
    return rhs;
}

inline float dot(const vec& lhs, const vec& rhs) {
    float res = 0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        res += lhs[i] * rhs[i];
    }
    return res;
}
