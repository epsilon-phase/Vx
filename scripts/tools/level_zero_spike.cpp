// A placed-kernel launch over Level Zero: the smallest thing that proves a
// payload image can be loaded and run on an Intel GPU by this machine, plus the
// argument cases a row or column view produces.
//
// What it is for. The dispatch library needs four things that only hardware can
// answer: whether a SPIR-V image from the toolchain loads, whether the flattened
// argument list of a Vx rank-2 tensor is what a kernel actually wants (two
// pointers, an offset, two sizes, two strides -- seven values, each its own
// argument), whether that list still describes the right elements when the
// tensor is a VIEW rather than a whole tensor, and whether the numbers come
// back. This program answers all four with a kernel that is known good, so a
// failure here is a failure of this program, not of Vx's device image.
//
// The three cases are the shapes a `spawn` body addresses:
//
//   whole   offset 0, sizes [8, 4], strides [4, 1] -- every element
//   row     offset 8, sizes [1, 4], strides [4, 1] -- the third row
//   column  offset 1, sizes [8, 1], strides [4, 1] -- the second column, which
//           is the non-unit-stride case: seven elements sit four apart
//
// Each case says which elements it must change, as a list, rather than
// recomputing the kernel's formula: a test that derives its expectation the way
// the code under test does passes for the wrong reason.
//
// Build and run: scripts/tools/level_zero_spike.sh
//
// Memory. The card is shared, and this machine gets unstable above 12 GiB of
// VRAM, so the program states what it allocates and refuses to exceed a ceiling
// (`VX_VRAM_CEILING`, default 12 GiB). It allocates 128 bytes. Level Zero in
// this version has no query for free device memory -- `zeDeviceGetMemoryProperties`
// reports static properties -- so the ceiling is enforced on this program's own
// allocation, not on the device's state.
//
// Nothing here uses oneAPI: Level Zero is the loader and the driver, and a
// C++ compiler with the headers is enough. SYCL arrives with oneMKL, where the
// vendor matmul library needs it.

#include <level_zero/ze_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define L0_CHECK(call)                                                     \
  do {                                                                     \
    ze_result_t r_ = (call);                                               \
    if (r_ != ZE_RESULT_SUCCESS) {                                          \
      std::fprintf(stderr, "%s:%d: %s failed with 0x%x\n", __FILE__,        \
                   __LINE__, #call, static_cast<unsigned>(r_));            \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

static std::vector<char> readFile(const char *path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(1);
  }
  return std::vector<char>((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
}

/// The description of the tensor a launch is handed: the seven values the
/// kernel takes as separate arguments.
struct TensorArgs {
  const char *name;
  uint64_t offset;
  uint64_t n0, n1;
  uint64_t s0, s1;
};

constexpr uint64_t kRows = 8, kCols = 4, kElems = kRows * kCols;
constexpr uint64_t kGroup = 8;
constexpr uint64_t kBytes = kElems * sizeof(float);

/// The largest amount of device memory this machine is safe to fill. Above it
/// the system becomes unstable, so the check is here rather than in a comment.
static uint64_t vramCeiling() {
  const char *env = std::getenv("VX_VRAM_CEILING");
  if (!env)
    return 12ull * 1024 * 1024 * 1024;
  return std::strtoull(env, nullptr, 10);
}

int main(int argc, char **argv) {
  const char *spvPath = argc > 1 ? argv[1] : "add_one.spv";
  const char *kernelName = argc > 2 ? argv[2] : "add_one";

  if (kBytes > vramCeiling()) {
    std::fprintf(stderr,
                 "refusing to run: %llu bytes allocated, ceiling %llu\n",
                 (unsigned long long)kBytes,
                 (unsigned long long)vramCeiling());
    return 1;
  }

  std::vector<char> image = readFile(spvPath);
  // The magic number is the cheapest way to catch "the file is not an image"
  // before the loader says something less direct.
  if (image.size() < 4 || std::memcmp(image.data(), "\x03\x02\x23\x07", 4) != 0) {
    std::fprintf(stderr, "%s is not a SPIR-V module\n", spvPath);
    return 1;
  }

  L0_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  uint32_t driverCount = 0;
  L0_CHECK(zeDriverGet(&driverCount, nullptr));
  std::vector<ze_driver_handle_t> drivers(driverCount);
  L0_CHECK(zeDriverGet(&driverCount, drivers.data()));

  // The first GPU any driver offers. This machine has one discrete Intel GPU
  // and an integrated AMD GPU that Level Zero does not describe, so the search
  // is what makes the program work wherever it runs.
  ze_driver_handle_t driver = nullptr;
  ze_device_handle_t device = nullptr;
  for (ze_driver_handle_t d : drivers) {
    uint32_t count = 0;
    L0_CHECK(zeDeviceGet(d, &count, nullptr));
    std::vector<ze_device_handle_t> devices(count);
    L0_CHECK(zeDeviceGet(d, &count, devices.data()));
    for (ze_device_handle_t dev : devices) {
      ze_device_properties_t props = {};
      props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      L0_CHECK(zeDeviceGetProperties(dev, &props));
      if (props.type == ZE_DEVICE_TYPE_GPU) {
        driver = d;
        device = dev;
        std::printf("device: %s  (%u compute units)\n", props.name,
                    props.numEUsPerSubslice * props.numSubslicesPerSlice *
                        props.numSlices);
        break;
      }
    }
    if (device)
      break;
  }
  if (!device) {
    std::fprintf(stderr, "no Level Zero GPU found\n");
    return 1;
  }

  // What the card holds, and what this program is allowed to take of it. The
  // total is static information; how much is free is not readable here.
  uint32_t moduleCount = 0;
  L0_CHECK(zeDeviceGetMemoryProperties(device, &moduleCount, nullptr));
  std::vector<ze_device_memory_properties_t> mem(moduleCount);
  for (auto &m : mem) {
    m.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
  }
  L0_CHECK(zeDeviceGetMemoryProperties(device, &moduleCount, mem.data()));
  for (const auto &m : mem)
    std::printf("memory module %s: %.2f GiB total\n", m.name,
                m.totalSize / (1024.0 * 1024.0 * 1024.0));
  std::printf("allocating %llu bytes of device memory (ceiling %.1f GiB)\n",
              (unsigned long long)kBytes, vramCeiling() / (1024.0 * 1024.0 * 1024.0));

  ze_context_desc_t contextDesc = {};
  contextDesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ze_context_handle_t context = nullptr;
  L0_CHECK(zeContextCreate(driver, &contextDesc, &context));

  ze_command_queue_desc_t queueDesc = {};
  queueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  ze_command_queue_handle_t queue = nullptr;
  L0_CHECK(zeCommandQueueCreate(context, device, &queueDesc, &queue));

  ze_command_list_desc_t listDesc = {};
  listDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  ze_command_list_handle_t list = nullptr;
  L0_CHECK(zeCommandListCreate(context, device, &listDesc, &list));

  ze_device_mem_alloc_desc_t allocDesc = {};
  allocDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  void *deviceBuffer = nullptr;
  L0_CHECK(zeMemAllocDevice(context, &allocDesc, kBytes, /*alignment=*/8, device,
                            &deviceBuffer));

  ze_module_desc_t moduleDesc = {};
  moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  moduleDesc.inputSize = image.size();
  moduleDesc.pInputModule =
      reinterpret_cast<const uint8_t *>(image.data());
  ze_module_handle_t module = nullptr;
  ze_module_build_log_handle_t buildLog = nullptr;
  ze_result_t moduleResult =
      zeModuleCreate(context, device, &moduleDesc, &module, &buildLog);
  if (moduleResult != ZE_RESULT_SUCCESS) {
    size_t logSize = 0;
    if (buildLog) {
      zeModuleBuildLogGetString(buildLog, &logSize, nullptr);
      std::vector<char> log(logSize + 1, 0);
      zeModuleBuildLogGetString(buildLog, &logSize, log.data());
      std::fprintf(stderr, "the driver refused the module: %s\n", log.data());
    } else {
      std::fprintf(stderr, "zeModuleCreate failed with 0x%x\n",
                   static_cast<unsigned>(moduleResult));
    }
    return 1;
  }

  ze_kernel_desc_t kernelDesc = {};
  kernelDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernelDesc.pKernelName = kernelName;
  ze_kernel_handle_t kernel = nullptr;
  L0_CHECK(zeKernelCreate(module, &kernelDesc, &kernel));
  L0_CHECK(zeKernelSetGroupSize(kernel, kGroup, 1, 1));

  // The tensor as the host sees it: 0, 1, 2, ... 31, then one addition per case
  // that covers that element. `touched` names the elements each case must reach,
  // and the lists are written out rather than derived, so a wrong stride in the
  // kernel cannot agree with a wrong stride here.
  std::vector<float> host(kElems);
  for (uint64_t i = 0; i < kElems; ++i)
    host[i] = static_cast<float>(i);

  const TensorArgs cases[] = {
      {"whole", 0, kRows, kCols, kCols, 1},
      {"row", 8, 1, kCols, kCols, 1},
      {"column", 1, kRows, 1, kCols, 1},
  };
  const std::vector<uint64_t> touched[3] = {
      {},                                          // whole: computed below
      {8, 9, 10, 11},                              // row 2
      {1, 5, 9, 13, 17, 21, 25, 29},               // column 1
  };
  std::vector<float> expected(kElems);
  for (uint64_t i = 0; i < kElems; ++i)
    expected[i] = static_cast<float>(i) + 1.0f;  // "whole" covers all of them

  int failures = 0;
  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
    const TensorArgs &a = cases[c];
    if (c) {
      for (uint64_t i : touched[c])
        expected[i] += 1.0f;
    }

    uint64_t offset = a.offset, n0 = a.n0, n1 = a.n1, s0 = a.s0, s1 = a.s1;
    void *allocated = deviceBuffer, *aligned = deviceBuffer;
    L0_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &allocated));
    L0_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void *), &aligned));
    L0_CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(uint64_t), &offset));
    L0_CHECK(zeKernelSetArgumentValue(kernel, 3, sizeof(uint64_t), &n0));
    L0_CHECK(zeKernelSetArgumentValue(kernel, 4, sizeof(uint64_t), &n1));
    L0_CHECK(zeKernelSetArgumentValue(kernel, 5, sizeof(uint64_t), &s0));
    L0_CHECK(zeKernelSetArgumentValue(kernel, 6, sizeof(uint64_t), &s1));

    L0_CHECK(zeCommandListAppendMemoryCopy(list, deviceBuffer, host.data(), kBytes,
                                           nullptr, 0, nullptr));
    L0_CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    // Ceiling, not truncation. Four elements in a group of eight is one group
    // with four idle work-items, not zero groups: truncating here meant the row
    // case never launched, and the mismatches read like a wrong stride.
    const uint64_t elems = a.n0 * a.n1;
    ze_group_count_t groups = {
        static_cast<uint32_t>((elems + kGroup - 1) / kGroup), 1, 1};
    L0_CHECK(zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr, 0,
                                             nullptr));
    L0_CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    L0_CHECK(zeCommandListAppendMemoryCopy(list, host.data(), deviceBuffer, kBytes,
                                           nullptr, 0, nullptr));
    L0_CHECK(zeCommandListClose(list));
    L0_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    L0_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    L0_CHECK(zeCommandListReset(list));

    uint64_t wrong = 0;
    for (uint64_t i = 0; i < kElems; ++i) {
      if (host[i] != expected[i]) {
        if (wrong < 3)
          std::printf("  %s: element %llu: got %f, wanted %f\n", a.name,
                       (unsigned long long)i, static_cast<double>(host[i]),
                       static_cast<double>(expected[i]));
        ++wrong;
      }
    }
    std::printf("case %-7s offset %llu sizes [%llu, %llu] strides [%llu, %llu]: %s\n",
                a.name, (unsigned long long)a.offset, (unsigned long long)a.n0,
                (unsigned long long)a.n1, (unsigned long long)a.s0,
                (unsigned long long)a.s1, wrong ? "WRONG" : "correct");
    failures += wrong ? 1 : 0;
  }
  return failures ? 1 : 0;
}
