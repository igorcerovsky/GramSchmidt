#ifndef _UTIL_H__
#define _UTIL_H__

#include <type_traits>
#include <array>
#include <random>
#include<functional>
#include<chrono>


namespace mtrx_impl
{
  template<typename T, typename U>
  using Common_type = typename std::common_type<T, U>::type;


  template<bool B, typename T = void>
  using Enable_if = typename std::enable_if<B, T>::type;


  template<typename T>
  class RandReal
  {
  public:
    RandReal(T low, T high) : dist{ low, high } {};
    T operator()() { return r(); }
  protected:
    std::default_random_engine re;
    std::uniform_real_distribution<T> dist;
    std::function<T()> r = std::bind(dist, re);
  };


  template<typename TimeT = std::chrono::milliseconds>
  struct MeasureTime
  {
    template<typename F, typename ...Args>
    static typename TimeT::rep run(F func, Args&&... args)
    {
      auto start = std::chrono::high_resolution_clock::now();
      func(std::forward<Args>(args)...);
      auto duration = std::chrono::duration_cast< TimeT>(std::chrono::high_resolution_clock::now() - start);

      return duration.count();
    }
  };


  template<typename TimeT = std::chrono::milliseconds>
  typename TimeT::rep Measure(std::function<void()> func)
  {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto duration = std::chrono::duration_cast<TimeT>(std::chrono::high_resolution_clock::now() - start);
    return duration.count();
  }


  template<typename List>
  bool check_non_jagged(const List& list)
  {
    auto i = list.begin();
    for (auto j = i + 1; j != list.end(); ++j)
      if (i->size() != j->size())
        return false;
    return true;
  }


  template<size_t N, typename List>
  std::array<size_t, N> derive_extents(const List& list)
  {
    std::array<size_t, N> a;
    auto f = a.begin();
    add_extents<N>(f, list);
    return a;
  }

  
  template<size_t N, typename I, typename List>
  Enable_if<(N>1), void> add_extents(I& first, const List& list)
  {
     assert(check_non_jagged(list));
    *first = list.size();
    add_extents<N-1>(++first, *list.begin());
  }


  template<size_t N, typename I, typename List>
  Enable_if<(N == 1), void> add_extents(I& first, const List& list)
  {
    *(first++) = list.size(); // we reached the deepest nesting
  }

} // namespace mtrx_impl

#endif // _UTIL_H__
