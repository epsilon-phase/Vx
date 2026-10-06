// A real GEMM on the Intel GPU through oneMKL, with its numbers checked and its
// speed measured.
//
// What it is for. The plan routes `kind=matmul` to oneMKL rather than to a
// generated kernel, and this is that route doing a large amount of work on this
// machine: a square fp32 GEMM, verified against a host reference at a size small
// enough to compute twice, then run at a size where the timing means something.
//
// It also puts a number on the card, which nothing before this did: the largest
// thing run through Vx's own launch path so far is 32 floats. That number is
// what a Vx-generated kernel will be compared against, and what says whether an
// Intel target is worth the work at all.
//
// Memory. The card is shared and this system is unstable above 12 GiB of VRAM in
// use, so the program states its footprint and refuses to allocate past a
// ceiling (`VX_VRAM_CEILING`, 12 GiB by default). Three fp32 matrices of the
// timed size: 786 MiB at 8192, which is the default.
//
// Build and run: scripts/tools/onemkl_gemm_spike.sh

#include <sycl/sycl.hpp>

#include <oneapi/mkl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace mkl = oneapi::mkl;

static constexpr uint64_t kMiB = 1024 * 1024;

static uint64_t vramCeiling() {
  const char *env = std::getenv("VX_VRAM_CEILING");
  if (!env)
    return 12ull * 1024 * 1024 * 1024;
  return std::strtoull(env, nullptr, 10);
}

/// One square GEMM of `n` by `n`, fp32, on the device.
static void gemm(sycl::queue &q, uint64_t n, float *a, float *b, float *c) {
  const float alpha = 1.0f, beta = 0.0f;
  auto e = mkl::blas::row_major::gemm(q, mkl::transpose::nontrans,
                                      mkl::transpose::nontrans, n, n, n, alpha,
                                      a, n, b, n, beta, c, n);
  e.wait();
}

int main(int argc, char **argv) {
  const uint64_t verify_n = 256;
  const uint64_t timed_n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 8192;
  const uint64_t timed_bytes = 3 * timed_n * timed_n * sizeof(float);

  sycl::device dev{sycl::gpu_selector_v};
  std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
  std::printf("device memory: %llu MiB\n",
              (unsigned long long)(dev.get_info<sycl::info::device::global_mem_size>() /
                                   kMiB));
  std::printf("timed GEMM: %llu x %llu, three matrices, %llu MiB total (ceiling %llu MiB)\n",
              (unsigned long long)timed_n, (unsigned long long)timed_n,
              (unsigned long long)(timed_bytes / kMiB),
              (unsigned long long)(vramCeiling() / kMiB));
  if (timed_bytes > vramCeiling()) {
    std::fprintf(stderr, "refusing to run: over the VRAM ceiling\n");
    return 1;
  }

  sycl::queue q{dev, sycl::property::queue::in_order{}};

  // Check the answer before timing it. A 256-square GEMM is 16.7 MFLOP on the
  // device and the same again on the host, which is cheap enough to do twice and
  // small enough that a wrong answer is obvious.
  {
    const uint64_t n = verify_n;
    std::vector<float> ha(n * n), hb(n * n), hc(n * n);
    for (uint64_t i = 0; i < n * n; ++i) {
      // The cast has to come before the subtraction: `i` is unsigned, so
      // `(i % 17) - 8` wraps to about 2^64 for the first eight values, and the
      // device then multiplies those instead of +-1. That is what "inf" came
      // from the first time this ran.
      ha[i] = static_cast<float>(static_cast<int64_t>(i % 17) - 8) / 8.0f;
      hb[i] = static_cast<float>(static_cast<int64_t>(i % 13) - 6) / 6.0f;
    }
    float *a = sycl::malloc_device<float>(n * n, q);
    float *b = sycl::malloc_device<float>(n * n, q);
    float *c = sycl::malloc_device<float>(n * n, q);
    q.memcpy(a, ha.data(), n * n * sizeof(float));
    q.memcpy(b, hb.data(), n * n * sizeof(float));
    q.memset(c, 0, n * n * sizeof(float)).wait();

    // What the device holds before the multiply. Without this, a wrong call and
    // a broken copy look the same from the answer, and only one of them is a
    // oneMKL problem.
    {
      float check[3] = {0, 0, 0};
      q.memcpy(check, a, sizeof(check)).wait();
      std::printf("  a[0..2] on the device: %f %f %f\n",
                  static_cast<double>(check[0]), static_cast<double>(check[1]),
                  static_cast<double>(check[2]));
    }

    gemm(q, n, a, b, c);
    q.memcpy(hc.data(), c, n * n * sizeof(float)).wait();

    // The reference in double, so the comparison measures the device and not the
    // host's accumulation order.
    double worst = 0.0, worst_want = 0.0;
    uint64_t worst_i = 0, worst_j = 0;
    for (uint64_t i = 0; i < n; ++i) {
      for (uint64_t j = 0; j < n; ++j) {
        double want = 0.0;
        for (uint64_t k = 0; k < n; ++k)
          want += static_cast<double>(ha[i * n + k]) * hb[k * n + j];
        const double got = static_cast<double>(hc[i * n + j]);
        const double err = std::fabs(got - want) / std::max(1.0, std::fabs(want));
        if (err > worst) {
          worst = err;
          worst_i = i;
          worst_j = j;
          worst_want = want;
        }
      }
    }
    // When it is wrong, say how: "inf" alone says the device wrote something
    // that is not a number, and where it did says whether the call was wrong or
    // the indexing was.
    if (!(worst < 1e-4))
      std::printf("  worst at [%llu][%llu]: got %f, wanted %f; first elements "
                  "of the result: %f %f %f\n",
                  (unsigned long long)worst_i, (unsigned long long)worst_j,
                  static_cast<double>(hc[worst_i * n + worst_j]), worst_want,
                  static_cast<double>(hc[0]), static_cast<double>(hc[1]),
                  static_cast<double>(hc[2]));
    std::printf("verify %llu x %llu: worst relative error %.3g -- %s\n",
                (unsigned long long)n, (unsigned long long)n, worst,
                worst < 1e-4 ? "correct" : "WRONG");
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    if (!(worst < 1e-4))
      return 1;
  }

  // Now the size where the timing means something. The buffers are filled with
  // ones rather than left as they were: denormals and NaNs would make the run
  // measure something other than the multiplication.
  const uint64_t n = timed_n;
  float *a = sycl::malloc_device<float>(n * n, q);
  float *b = sycl::malloc_device<float>(n * n, q);
  float *c = sycl::malloc_device<float>(n * n, q);
  q.fill(a, 1.0f, n * n).wait();
  q.fill(b, 1.0f, n * n).wait();
  q.fill(c, 0.0f, n * n).wait();

  gemm(q, n, a, b, c);  // warm-up: the first call pays for kernel selection
  double best = 1e30, total = 0.0;
  const int runs = 5;
  for (int r = 0; r < runs; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    gemm(q, n, a, b, c);
    auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    best = std::min(best, ms);
    total += ms;
  }
  const double flop = 2.0 * static_cast<double>(n) * static_cast<double>(n) *
                      static_cast<double>(n);
  std::printf("%llu x %llu x %llu: best %.1f ms, mean %.1f ms, %.2f TFLOPS\n",
              (unsigned long long)n, (unsigned long long)n, (unsigned long long)n,
              best, total / runs, flop / (best / 1000.0) / 1e12);

  // One element of the result is n: a column of A dotted with a row of B, all
  // ones. Checking it costs nothing and catches a kernel that did not run.
  float corner = 0.0f;
  q.memcpy(&corner, c, sizeof(float)).wait();
  std::printf("c[0][0] = %f (expected %llu) -- %s\n", static_cast<double>(corner),
              (unsigned long long)n,
              corner == static_cast<float>(n) ? "correct" : "WRONG");

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
  return corner == static_cast<float>(n) ? 0 : 1;
}
