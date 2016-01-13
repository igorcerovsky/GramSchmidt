// Avx1.cpp : Defines the entry point for the console application.
//

#include <immintrin.h>
#define USE_VECTOR
#ifdef USE_VECTOR
#include <vector>
#include <typeinfo>
template<typename T>
using vec = std::vector<T>;
#else
#include <valarray>
template<typename T>
using vec = std::valarray<T>;
#endif

#include <iostream>

#include "../GramSchmidt/Logger.h"
#include "../GramSchmidt/util.h"


//#define TRACE_RESULT


Logger logger{ "dot_avx_log.txt" };

double DotAvx(const vec<__m256d>& u, const vec<__m256d>& v, double& init)
{
  constexpr size_t sz = 4;
  register __m256d res = _mm256_setzero_pd();
  for (size_t k = 0; k < v.size(); ++k)
  {
    __m256d tmp = _mm256_mul_pd(u[k], v[k]);
    res = _mm256_hadd_pd(res, tmp);
  }

  for (size_t i = 0; i < sz; ++i)
    init += res.m256d_f64[i];

  return init;
}

double DotBase(const vec<double>& u, const vec<double>& v, double& init)
{
#define SIMPLE
#ifndef SIMPLE
  constexpr size_t sz = 4;
  register T tmp[]{ 0,0,0,0 };
  for (size_t k = 0; k < v.size(); ++k/*k+=4*/)
  {
    tmp[0] += u[k+0] * v[k+0];
    tmp[1] += u[k+1] * v[k+1];
    tmp[2] += u[k+2] * v[k+2];
    tmp[3] += u[k+3] * v[k+3];
  }
  init = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#else
  constexpr size_t sz = 4;
  register double tmp{ 0 };
  for (size_t k = 0; k < v.size(); ++k)
  {
    tmp += u[k] * v[k];
  }
  init = tmp;
#endif

  return init;
}


double DotAvx(const vec<double>& u, const vec<double>& v, double& init)
{
  constexpr size_t offset = 4;
  register __m256d x;
  register __m256d y;

  register __m256d res = _mm256_setzero_pd();
  const size_t sz = v.size() / offset;
  double* p_u = const_cast<double*>(u.data());
  double* p_v = const_cast<double*>(v.data());
  for (size_t k = 0; k < sz; ++k)
  {
    x = _mm256_load_pd(p_u);
    p_u += offset;

    y = _mm256_load_pd(p_v);
    p_v += offset;

    register __m256d temp = _mm256_mul_pd(x, y);
    res = _mm256_hadd_pd(res, temp);
  }
  for (size_t i = 0; i < offset; ++i)
    init += res.m256d_f64[i];

  return init;
}


template<typename T, typename F>
void RunMeasure(F fnc, const vec<T>& u, const vec<T>& v)
{
  using measure_time = mtrx_impl::MeasureTime<std::chrono::nanoseconds>;
  double sum{ 0 };
  auto a = measure_time::run<F>(fnc, u, v, sum);

#ifdef TRACE_RESULT
  logger << a << " {" << sum << "}  ";
#else
  logger << a << " ";
#endif
}


template <typename T, typename = std::enable_if_t<std::is_floating_point<T>::value> >
void RandData(vec<T>& v)
{
  mtrx_impl::RandReal<T> rnd{ -100, 100 };
  for (size_t i = 0; i < v.size(); ++i)
  {
      v[i] = rnd();
  }
}

void FixedInit(vec<double>& u, vec<double>& v)
{
  u = vec<double>{ 1,2,3,4, 1,1,3,4, 1,1,1,4, 1,1,2,2 };
  v = vec<double>{ 4,5,6,7, 4,5,6,7, 4,5,6,7, 4,5,6,7 };
}

template<class T>
constexpr size_t Scale()
{
  return std::is_floating_point<T>::value == true ? 4 : 1;
}


//template<typename T>
//void TestFixed()
//{
//  constexpr size_t scale = Scale<T>();
//  using value_type = T;
//  using vvv = vec<value_type>;
//
//  logger << typeid(T).name() << "\n";
//  vvv u, v;
//  FixedInit(u, v);
//  std::cout << v.size() / scale << "  ";
//
//  using test_fnc = double(const vvv&, const vvv&, double&);
//  RunMeasure<value_type, test_fnc>(DotAvx, u, v);
//
//  logger << "\n";
//}

void Test()
{
  constexpr size_t align = 4;
  using vec_d = vec<double>;
  using vec_m = vec<__m256d>;

  logger << "size time_base time_axv_dbl time_avx_m256d\n";
  for (size_t k = 10; k < 10'000'000; k *= 2)
  {
    const size_t sz = k*align;
    vec_d u(sz);
    vec_d v(sz);
    RandData(u);
    RandData(v);
    logger << sz << " ";

    using test_fnc_d = double(const vec_d&, const vec_d&, double&);
    RunMeasure<double, test_fnc_d>(DotBase, u, v);
    RunMeasure<double, test_fnc_d>(DotAvx, u, v);

    vec_m um(k);
    vec_m vm(k);
    for (size_t i = 0; i < k; ++i)
    {
      for (size_t j : {0, 1, 2, 3})
      {
        um[i].m256d_f64[j] = u[i*align + j];
        vm[i].m256d_f64[j] = v[i*align + j];
      }
    }
    using test_fnc_m = double(const vec_m&, const vec_m&, double&);
    RunMeasure<__m256d, test_fnc_m>(DotAvx, um, vm);

    logger  << "\n";
  }
}

int main()
{
  Test();
  logger << "\n";

  return 0;
}

