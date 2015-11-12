#pragma once
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>

/**
 * Small linear algebra library that supports basic operations between vectors: difference, addition or dot product of 
 * two vectors, and multiplication and division by a scalar.
 * 
 * This module uses expression templates (https://en.wikipedia.org/wiki/Expression_templates) to perform efficient vector operations.
 * 
 * Vector operations like: (u + alpha * v) use a single for loop, while
 * naive operator overloading would use two loops.
 * 
 * Examples:
 * Vec v1({2,0,2});
 * v1 /= 2;
 * Vec v2({1,0,0});
 * Vec v = 0.5 * (v1 + v2);
 * float u = v1.dot(v2);
 * std::cout << v << std::endl;    #[1, 0, 0.5]
 * 
 * TODO: integrate with BLAS
 */

template <typename E>
class VecExpression {
public:
    typedef std::vector<float> container_type;
    typedef typename container_type::size_type size_type;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;

    size_type size() const { return static_cast<E const&>(*this).size(); }
    value_type operator[](size_type i) const { return static_cast<E const&>(*this)[i]; }

    operator E&() { return static_cast<E&>(*this); }
    operator E const&() const { return static_cast<const E&>(*this); }
};

class Vec : public VecExpression<Vec> {
    container_type _data;
public:
    reference operator[](size_type i) { return _data[i]; }
    value_type operator[](size_type i) const { return _data[i]; }
    size_type size() const { return _data.size(); }

    Vec() {}
    Vec(size_type n) : _data(n) {}
    Vec(size_type n, float val) : _data(n, val) {}
    Vec(Vec::container_type v) : _data(v) {}

    friend std::ostream& operator<<(std::ostream &o, Vec const& self) {
        std::ostringstream ss;
        if (!self._data.empty()) {
            std::copy(self._data.begin(), self._data.end() - 1, std::ostream_iterator<int>(ss, ","));
            ss << self._data.back();
        }
        return o << "[" << ss.str() << "]";
    }
    
    template <typename E>
    Vec(VecExpression<E> const& vec) {
        E const& v = vec;
        _data.resize(v.size());
        for (size_type i = 0; i != v.size(); ++i) {
            _data[i] = v[i];
        }
    }

    template <typename E>
    void operator=(VecExpression<E> const& vec) {
        E const& v = vec;
        _data.resize(v.size());
        for (size_type i = 0; i != v.size(); ++i) {
            _data[i] = v[i];
        }
    }

    template <typename E>
    float dot(VecExpression<E> const& vec) const {
        E const& v = vec;
        float x = 0;
        for (size_type i = 0; i != v.size(); ++i) {
            x += _data[i] * v[i];
        }
        return x;
    }

    template <typename E>
    void operator+=(VecExpression<E> const& vec) {
        *this = *this + vec;
    }

    template <typename E>
    void operator-=(VecExpression<E> const& vec) {
        *this = *this - vec;
    }

    void operator*=(float alpha) {
        for (size_type i = 0; i != size(); ++i) {
            _data[i] *= alpha;
        }
    }

    void operator/=(float alpha) {
        for (size_type i = 0; i != size(); ++i) {
            _data[i] /= alpha;
        }
    }
    
    float norm() const {
        float res = 0;
        for (size_type i = 0; i != size(); ++i) {
            res += pow(_data[i], 2);
        }
        return sqrt(res);
    }
    
    const value_type* data() const { return _data.data(); }
    value_type* data() { return _data.data(); }
};

template <typename E1, typename E2>
class VecDifference : public VecExpression<VecDifference<E1, E2>> {
    E1 const& u;
    E2 const& v;
public:
    typedef Vec::size_type size_type;
    typedef Vec::value_type value_type;
    VecDifference(VecExpression<E1> const& u, VecExpression<E2> const& v) : u(u), v(v) {}
    size_type size() const { return v.size(); }
    value_type operator[](Vec::size_type i) const { return u[i] - v[i]; }
};

template <typename E1, typename E2>
class VecAddition : public VecExpression<VecAddition<E1, E2>> {
    E1 const& u;
    E2 const& v;
public:
    typedef Vec::size_type size_type;
    typedef Vec::value_type value_type;
    VecAddition(VecExpression<E1> const& u, VecExpression<E2> const& v) : u(u), v(v) {}
    size_type size() const { return v.size(); }
    value_type operator[](Vec::size_type i) const { return u[i] + v[i]; }
};

template <typename E>
class VecMinus : public VecExpression<VecMinus<E>> {
    E const& u;
public:
    typedef Vec::size_type size_type;
    typedef Vec::value_type value_type;
    VecMinus(VecExpression<E> const& u) : u(u) {}
    size_type size() const { return u.size(); }
    value_type operator[](Vec::size_type i) const { return -u[i]; }
};

template <typename E>
class VecScaled : public VecExpression<VecScaled<E>> {
    float alpha;
    E const& v;
public:
    VecScaled(float alpha, VecExpression<E> const& v) : alpha(alpha), v(v) {}
    Vec::size_type size() const { return v.size(); }
    Vec::value_type operator[](Vec::size_type i) const { return alpha * v[i]; }
};

template <typename E1, typename E2>
VecAddition<E1, E2> const
operator+(VecExpression<E1> const& u, VecExpression<E2> const& v) {
    return VecAddition<E1, E2>(u, v);
}

template <typename E>
VecMinus<E> const
operator-(VecExpression<E> const& u) {
    return VecMinus<E>(u);
}

template <typename E1, typename E2>
VecDifference<E1, E2> const
operator-(VecExpression<E1> const& u, VecExpression<E2> const& v) {
    return VecDifference<E1, E2>(u, v);
}

template <typename E>
VecScaled<E> const
operator*(float alpha, VecExpression<E> const& v) {
    return VecScaled<E>(alpha, v);
}

template <typename E>
VecScaled<E> const
operator*(VecExpression<E> const& v, float alpha) {
    return VecScaled<E>(alpha, v);
}

template <typename E>
VecScaled<E> const
operator/(VecExpression<E> const& v, float alpha) {
    return VecScaled<E>(1 / alpha, v);
}
