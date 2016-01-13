#include <iostream>

#include <string>
#include "matrix.h"
#include "matrix_out.h"
#include "Logger.h"

template<typename T>
void Test()
{
  using namespace mtrx;
  using namespace mtrx_impl;
  using std::cout;

  using value_type = T;
  using Mat = Matrix<value_type, Orient::col_major>;

  Logger log{ "test.log" };
  RandReal<T> rnd{ -100, 100 };
  std::array < size_t, 10> sizes{ 6, 10, 20, 40, 80, 160, 320, 640, 800, 1'000 };

  using gs_fnc = void(*)(Mat &);
  std::array<gs_fnc, 3> fncs{ gram_schmidt_inplace<value_type>,
    gram_schmidt_inplace_eliminate_temporary_copy<value_type>,
    gram_schmidt_inplace_naive<value_type> };

  log << R"("Computing Gram-Schmidt results in microseconds...")" << "\n" <<
    "size[rows,cols] cols _inplace  _eliminate_temporary_copy  _naive" << "\n";
  for (auto sz : sizes)
  {
    Mat T(sz, sz);
    RandomMatrix(T, rnd );

    log << R"("[)" << sz << ", " << sz << R"(]" )" << sz << "  ";
    for (auto a : fncs)
    {
      Mat M{ T };
      auto a1 = MeasureTime<std::chrono::microseconds>::run<void(Mat &)>(a, M);
      log << "  " << a1 << "  ";
    }
    log << "\n";
  }
}


int main()
{
  using namespace mtrx_impl;
  using namespace mtrx;
  using std::cout;
  using value_type = double;
  using Mat = Matrix<value_type, Orient::col_major>;
  
  Mat A{ { 1.f,1.f,0.f,0.f },{ 0.f,2.f,1.f,1.f } };
  gram_schmidt_inplace(A);
  cout << A;

  Mat M(2, 3);
  RandReal<value_type> rnd{ -10, 10 };
  RandomMatrix(M, rnd);
  gram_schmidt_inplace(M);
  cout << M;

  Mat N(3, 2);
  RandomMatrix(N, rnd);
  cout << N;

  Test<value_type>();

  return 0;
}