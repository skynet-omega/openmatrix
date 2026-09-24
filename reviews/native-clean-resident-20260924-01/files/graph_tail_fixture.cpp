#include <chrono>
#include <cstdlib>
#include <iostream>

extern "C" int native_tail_fixture(int limit, int* observed);
extern "C" const char* native_tail_error();

int main() {
  int observed = -1;
  const auto start = std::chrono::steady_clock::now();
  const int status = native_tail_fixture(7, &observed);
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  if (status != 0) {
    std::cerr << "FAIL: " << native_tail_error() << "\n";
    return EXIT_FAILURE;
  }
  std::cout << "PASS observed=" << observed << " wall_s=" << seconds << "\n";
  return observed == 7 ? EXIT_SUCCESS : EXIT_FAILURE;
}
