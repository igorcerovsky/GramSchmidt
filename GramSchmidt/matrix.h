#ifndef _MATRIX_H__
#define _MATRIX_H__

#include <vector>
#include <valarray>
#include <initializer_list>
#include <cassert>

#include "util.h"
#include "matrix_impl.h"

namespace mtrx {

  using std::initializer_list;
  using std::valarray;
  using std::slice_array;
  using std::array;
  using std::slice;
  using std::vector;

  template<typename T>
  using Vector = valarray<T>;

  enum class Orient { row_major, col_major };
  
  template<typename T, Orient O>
  class MatrixBase {
  public:
    using data_type = valarray<T>;
    using value_type = T;

    MatrixBase() = default;
    MatrixBase(MatrixBase&&) = default;
    MatrixBase& operator=(MatrixBase&&) = default;

    MatrixBase(MatrixBase const&) = default;
    MatrixBase& operator=(MatrixBase const&) = default;

    // do not allow simple initilization without dimensions
    MatrixBase(initializer_list<T>) = delete;
    MatrixBase& operator=(initializer_list<T>) = delete;

    ~MatrixBase() = default;

    MatrixBase(size_t rows, size_t cols)
    {
      this->sz = { rows,cols };
      for (size_t i = 0; i < this->sz[0]; ++i)
        elems.push_back(data_type(sz[1]));
    };

    Orient orientation() { return O; }

    size_t rows() { return sz[0]; }
    size_t rows() const { return sz[0]; }
    size_t cols() { return sz[1]; }
    size_t cols() const { return sz[1]; }

    size_t size() { return sz[0] * sz[1]; }
    size_t size() const { return sz[0] * sz[1]; }

    data_type& operator[](size_t n) { return elems[n]; }
    const data_type& operator[](size_t n) const { return elems[n]; }

  protected:
    array<size_t, 2> sz;
    vector<data_type> elems;
  };


  template<typename T, Orient O >
  class Matrix : public MatrixBase<T, O> 
  {};


  template<typename T >
  class Matrix<T, Orient::col_major> : public MatrixBase<T, Orient::col_major> {
  public:
    using data_type = valarray<T>;
    using init_list = initializer_list<initializer_list<T>>;

    Matrix() = default;
    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;

    Matrix(Matrix const&) = default;
    Matrix& operator=(Matrix const&) = default;

    Matrix(init_list list);
    Matrix& operator=(init_list) = delete;

    Matrix(size_t rows, size_t cols) 
    {
      this->sz = { rows,cols };
      for (size_t i = 0; i < this->sz[1]; ++i)
        elems.push_back(data_type(sz[0]));
    };

     ~Matrix() = default;

    T& operator()(size_t i, size_t j) { return elems[j][i]; }
    const T& operator()(size_t i, size_t j) const { return elems[j][i]; }

    template<typename T>
    friend Matrix<T, Orient::col_major> gram_schmidt(const Matrix<T, Orient::col_major>&);

    template<typename T>
    friend void gram_schmidt_inplace(Matrix<T, Orient::col_major>&);

  };
  

  template<typename T>
  Matrix<T, Orient::col_major>::Matrix(init_list list)
  {
    this->sz = mtrx_impl::derive_extents<2>(list);
    std::swap(sz[0], sz[1]);
    for (auto a : list)
      elems.push_back(a);
  }


  template<typename T>
  Matrix<T, Orient::col_major> gram_schmidt_naive(const Matrix<T, Orient::col_major>& m)
  {
    Matrix<T, Orient::col_major> A(m);
    Matrix<T, Orient::col_major> Q(m.rows(), m.cols());
    Matrix<T, Orient::col_major> R(m.rows(), m.cols());

    for (size_t k = 0; k < A.cols(); ++k)
    {
      R[k][k] = len(A[k]);
      Q[k] = A[k] / R[k][k];
      for (size_t j = k + 1; j < A.cols(); ++j)
      {
        R[k][j] = dot(Q[k], A[j]);
        A[j] -= R[k][j] * Q[k];
      }
    }
    return Q;
  }


  template<typename T>
  Matrix<T, Orient::col_major> gram_schmidt(const Matrix<T, Orient::col_major>& M)
  {
    Matrix<T, Orient::col_major> Q(M);
    gram_schmidt_inplace(Q);

    return Q;
  }


  template<typename T>
  void gram_schmidt_inplace(Matrix<T, Orient::col_major>& Q)
  {
    Matrix<T, Orient::col_major> A(Q);

    for (size_t k = 0; k < A.cols(); ++k)
    {
      std::memcpy(&Q[k][0], &A[k][0], Q[k].size()*sizeof(T)); // memcpy except operator =
      Q[k] /= len(A[k]);
      for (size_t j = k + 1; j < A.cols(); ++j)
      {
        std::memcpy(&A[k][0], &Q[k][0], A[k].size()*sizeof(T));
        A[k] *= dot(Q[k], A[j]);
        A[j] -= A[k];
      }
    }
  }


  template<typename T>
  void gram_schmidt_inplace_eliminate_temporary_copy(Matrix<T, Orient::col_major>& Q)
  {
    Matrix<T, Orient::col_major> A(Q);

    for (size_t k = 0; k < A.cols(); ++k)
    {
      Q[k] = A[k];
      Q[k] /= len(A[k]); // no temporary array copy
      for (size_t j = k + 1; j < A.cols(); ++j)
      {
        A[k] = Q[k];
        A[k] *= dot(Q[k], A[j]);
        A[j] -= A[k];
      }
    }
  }


  template<typename T>
  void gram_schmidt_inplace_naive(Matrix<T, Orient::col_major>& Q)
  {
    Matrix<T, Orient::col_major> A(Q);
    
    for (size_t k = 0; k < A.cols(); ++k)
    {
      Q[k] = A[k] / len(A[k]); // creates temporary array copy
      for (size_t j = k + 1; j < A.cols(); ++j)
      {
        A[j] -= Q[k] * dot(Q[k], A[j]); // also creates temporary array copy
        //A[j] = A[j] - Q[k] * dot(Q[k], A[j]); // also creates temporary array copy
      }
    }
  }


} // namespace mtrx

#endif // _MATRIX_H__
