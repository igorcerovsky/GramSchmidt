#ifndef _MATRIX_OUT_H__
#define _MATRIX_OUT_H__

#include "matrix.h"
#include "util.h"

namespace mtrx {

  template<typename T>
  std::ostream& operator<<(std::ostream& os, const Vector<T>& v)
  {
    for (auto a : v)
      os << a << " ";
    return os << "\n";
  }

  template<typename T, Orient O>
  std::ostream& operator<<(std::ostream& os, const Matrix<T, O>& m)
  {
    os << "matrix[" << m.rows() << "," << m.cols() << "]\n";
    for (size_t i = 0; i < m.rows(); ++i)
    {
      for (auto& a : m[i] )
        os << a << " ";
      os << "\n";
    }
    return os;
  }

  template<typename T>
  std::ostream& operator<<(std::ostream& os, const Matrix<T, Orient::col_major>& m)
  {
    os << "COLUMN MAJOR DATA ORIENTATION matrix[" << m.rows() << "," << m.cols() << "]\n";
    for (size_t i = 0; i < m.rows(); ++i)
    {
      for (size_t j = 0; j < m.cols(); ++j)
        os << m(i, j) << " ";
      os << "\n";
    }
    return os;
  }

  template<typename T, Orient O>
  void RandomMatrix( Matrix<T, O>& m, mtrx_impl::RandReal<T>& rnd)
  {
    for (size_t i = 0; i < m.rows(); ++i) {
      for (size_t j = 0; j < m.cols(); ++j) {
        m(i, j) = rnd();
      }
    }
  }

} // namespace mtrx

#endif // _MATRIX_OUT_H__
