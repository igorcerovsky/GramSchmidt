#include "stdafx.h"
#include "CppUnitTest.h"

#include "../GramSchmidt/matrix.h"
#include "../GramSchmidt/matrix_impl.h"
#include "../GramSchmidt/matrix_out.h"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace mtrx;
using namespace mtrx_impl;

namespace GramSchmidtTest
{		
	TEST_CLASS(UnitTestMatrix)
	{
	public:

    TEST_METHOD(TestMatrixOrientation)
    {
      MatrixBase<int, Orient::row_major> A;
      Assert::AreEqual(int(A.orientation()), int(Orient::row_major));

      MatrixBase<int, Orient::col_major> B;
      Assert::AreEqual(int(B.orientation()), int(Orient::col_major));
    }

    TEST_METHOD(TestMatrixColConstructorList)
    {
      using value_type = int;
      using Mat = Matrix<value_type, Orient::col_major>;
      Mat M{ { 11,21,31 },{ 12,22,32 } };

      Assert::AreEqual(11, M(0, 0));
      Assert::AreEqual(21, M(1, 0));
      Assert::AreEqual(31, M(2, 0));
      Assert::AreEqual(12, M(0, 1));
      Assert::AreEqual(22, M(1, 1));
      Assert::AreEqual(32, M(2, 1));
    }


    template<typename T>
    void EqualMatrixes(const Matrix<T, Orient::col_major>& Q, const Matrix<T, Orient::col_major>& R, const T tolerance)
    {
      for (size_t i = 0; i < Q.rows(); ++i)
      {
        for (size_t j = 0; j < Q.cols(); ++j)
        {
          Assert::AreEqual(Q(i, j), R(i, j), tolerance);
        }
      }
    }

    
    TEST_METHOD(TestMatrixConstructor)
    {
      using value_type = float;
      using Mat = Matrix<value_type, Orient::col_major>;
      Mat M(2,3);
      RandReal<value_type> rnd{ -100, 100 };
      RandomMatrix(M, rnd);

      Mat N{ M };
      EqualMatrixes<value_type>(M, N, value_type{ 0.0f });
    }


    template<typename T>
    void TestGramSchmidt()
    {
      using value_type = T;
      using MatrixFloat = Matrix<value_type, Orient::col_major>;
      MatrixFloat A{
        { 1, 1, 1 },
        { 2, 1, 0 },
        { 5, 1, 3 } };

      MatrixFloat S{
        { 0.57735f,    0.57735f,  0.57735f },
        { 0.707107f,       0.0f, -0.707107f },
        { 0.408248f, -0.816497f,  0.408248f } };

      MatrixFloat N = gram_schmidt_naive(A);
      EqualMatrixes<value_type>(N, S, value_type{ 1.0e-6f });

      MatrixFloat Q = gram_schmidt(A);
      EqualMatrixes<value_type>(Q, S, value_type{ 1.0e-6f });

      MatrixFloat D(A);
      gram_schmidt_inplace(D);
      EqualMatrixes<value_type>(D, S, value_type{ 1.0e-6f });
    }


    TEST_METHOD(TestGramSchmidt)
    {
      TestGramSchmidt<float>();
      TestGramSchmidt<double>();
    }


    template<typename T>
    void AreOrhogonalCols(const Matrix<T, Orient::col_major>& M)
    {
      for (size_t i = 1; i < M.cols(); ++i)
      {
        auto a = dot(M[i - 1], M[i]);
        Assert::AreEqual(a, T{ 0 }, std::numeric_limits<T>::epsilon()*10);
      }
    }

    template<typename T>
    void IsOrthogonalRandomMatrix(size_t rows, size_t cols)
    {
      using value_type = T;
      using Mat = Matrix<value_type, Orient::col_major>;
      Mat M(rows, cols);
      RandReal<value_type> rnd{ -1, 1 };
      RandomMatrix(M, rnd);
      gram_schmidt_inplace(M);
      AreOrhogonalCols(M);
    }

    TEST_METHOD(TestOrthogonality)
    {
      IsOrthogonalRandomMatrix<float>(11, 11);
      IsOrthogonalRandomMatrix<double>(11, 11);
      IsOrthogonalRandomMatrix<float>(4, 3);
      IsOrthogonalRandomMatrix<double>(4, 3);
    }

  };
}