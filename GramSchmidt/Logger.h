#ifndef _LOGGER_H__
#define _LOGGER_H__
#include <chrono>
#include <ctime>
#include <iomanip>
#include <fstream>


struct Logger {
  std::ofstream ofs;
  Logger(const std::string& file) : ofs(file, std::ofstream::out /*| std::ofstream::app*/) {
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    ofs << std::put_time(std::localtime(&now), "%F %T") << '\n';
  }
  ~Logger() { ofs << "\n\n"; }
  template<typename U>
  Logger& operator<<(U u) { ofs << u; std::cout << u;  return  *this; }
};

#endif // _LOGGER_H__
