// A placed-kernel launch over Level Zero: the smallest thing that proves a
// payload image can be loaded and run on an Intel GPU by this machine.
//
// What it is for. The dispatch library needs three things that only hardware
// can answer: whether a SPIR-V image from the toolchain loads, whether the
// flattened argument list of a Vx rank-2 tensor is what a kernel actually
// wants (two pointers, an offset, two sizes, two strides -- seven values, each
// its own argument), and whether the numbers the kernel writes come back. This
// program answers all three with a kernel that is known good, so a failure
// here is a failure of this program, not of Vx's device image.
//
// Build and run: scripts/tools/level_zero_spike.sh
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

int main(int argc, char **argv) {
  const char *spvPath = argc > 1 ? argv[1] : "add_one.spv";
  const char *kernelName = argc > 2 ? argv[2] : "add_one";

  // The tensor the kernel is given: 8 rows of 4 floats, contiguous, so the
  // row stride is 4 and the element stride is 1. The kernel is launched with
  // one work-item per element, in groups of 8.
  constexpr uint64_t kRows = 8, kCols = 4, kElems = kRows * kCols;
  constexpr uint64_t kGroup = 8;
  const size_t bytes = kElems * sizeof(float);

  std::vector<float> host(kElems);
  for (uint64_t i = 0; i < kElems; ++i)
    host[i] = static_cast<float>(i);

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
        std::printf("device: %s  (%u compute units, %.1f GiB max allocation)\n",
                    props.name,
                    props.numEUsPerSubslice * props.numSubslicesPerSlice *
                        props.numSlices,
                    props.maxMemAllocSize / (1024.0 * 1024.0 * 1024.0));
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
  L0_CHECK(zeMemAllocDevice(context, &allocDesc, bytes, /*alignment=*/8, device,
                            &deviceBuffer));

  L0_CHECK(zeCommandListAppendMemoryCopy(list, deviceBuffer, host.data(), bytes,
                                         nullptr, 0, nullptr));
  L0_CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

  // The image's parameters, in order: allocated, aligned, offset, n0, n1, s0,
  // s1. `allocated` and `aligned` are the same address here because the buffer
  // has no sub-view; a strided view would make them differ.
  uint64_t offset = 0, n0 = kRows, n1 = kCols, s0 = kCols, s1 = 1;

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
  L0_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &deviceBuffer));
  L0_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void *), &deviceBuffer));
  L0_CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(uint64_t), &offset));
  L0_CHECK(zeKernelSetArgumentValue(kernel, 3, sizeof(uint64_t), &n0));
  L0_CHECK(zeKernelSetArgumentValue(kernel, 4, sizeof(uint64_t), &n1));
  L0_CHECK(zeKernelSetArgumentValue(kernel, 5, sizeof(uint64_t), &s0));
  L0_CHECK(zeKernelSetArgumentValue(kernel, 6, sizeof(uint64_t), &s1));

  ze_group_count_t groups = {static_cast<uint32_t>(kElems / kGroup), 1, 1};
  L0_CHECK(
      zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr, 0, nullptr));
  L0_CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
  L0_CHECK(zeCommandListAppendMemoryCopy(list, host.data(), deviceBuffer, bytes,
                                         nullptr, 0, nullptr));
  L0_CHECK(zeCommandListClose(list));
  L0_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  L0_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  // The kernel adds one to every element through the flattened argument list.
  uint64_t wrong = 0;
  for (uint64_t i = 0; i < kElems; ++i) {
    const float want = static_cast<float>(i) + 1.0f;
    if (host[i] != want) {
      if (wrong < 4)
        std::fprintf(stderr, "element %llu: got %f, wanted %f\n",
                     (unsigned long long)i, static_cast<double>(host[i]),
                     static_cast<double>(want));
      ++wrong;
    }
  }
  std::printf("kernel %s on %llu elements: %s\n", kernelName,
              (unsigned long long)kElems,
              wrong ? "WRONG" : "correct");
  return wrong ? 1 : 0;
}
