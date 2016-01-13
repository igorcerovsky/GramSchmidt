#ifndef _MATRIX_IMPL_H__
#define _MATRIX_IMPL_H__

#include <algorithm>
#include <numeric>

using std::valarray;

namespace mtrx {

  template<typename T>
  inline T dot(const valarray<T>& v, const valarray<T>& u)
  {
    return std::inner_product(&v[0], &v[0] + v.size(), &u[0], T{ 0 });
  }

  template<typename T>
  T len(const valarray<T>& v)
  {
    return sqrt(dot(v, v));
  }

  template<typename T>
  void unit(valarray<T>& u)
  {
    u *= T{ 1 } / len<T>(u);
  }


  template<typename T>
  valarray<T> unit(const valarray<T>& u)
  {
    valarray<T> r = u;
    unit(r);
    return r;
  }

} // namespace mtrx

#endif // _MATRIX_IMPL_H__
