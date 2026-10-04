//===- payload_test.cpp - The dispatch payload's own format ----*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The payload a dispatch carries: the kernel name, then NUL-separated
// `key=value` entries. This checks the two that are about the format itself.
//
// `abi=` is the version, and it exists because the entry list grows. Every
// consumer walks past a key it was not asked for, which is what let `kind=`,
// `roles=`, `topo=` and the rest land one at a time without breaking anything
// -- but that protects a reader from a key it does not know, not from a value
// it misreads. The version is what a dispatch library checks before it trusts
// anything else.
//
// `imagebin=` carries a device image that is not text. `image=` is one
// NUL-terminated entry, which is exactly what a PTX module is and exactly what
// an SPIR-V module is not: the first word of SPIR-V holds NUL bytes, so a raw
// binary image would truncate the entry and every later one with it. Hence
// base64, and hence a decoder strict enough that a corrupted image is refused
// rather than silently shortened.
//
// Driven by tests/integration_test/payload_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../include/vx_hardware_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

/// Standard base64, written here rather than reusing the decoder: a round trip
/// through one implementation proves the two agree with each other, and a
/// decoder that is wrong in the same direction as its encoder is still wrong.
/// The published vectors below pin the decoder itself.
std::string b64(const unsigned char *data, size_t len) {
  static const char *kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (size_t i = 0; i < len; i += 3) {
    unsigned int v = (unsigned int)data[i] << 16;
    bool two = i + 1 < len;
    bool three = i + 2 < len;
    if (two)
      v |= (unsigned int)data[i + 1] << 8;
    if (three)
      v |= (unsigned int)data[i + 2];
    out.push_back(kAlphabet[(v >> 18) & 0x3F]);
    out.push_back(kAlphabet[(v >> 12) & 0x3F]);
    out.push_back(two ? kAlphabet[(v >> 6) & 0x3F] : '=');
    out.push_back(three ? kAlphabet[v & 0x3F] : '=');
  }
  return out;
}

/// A payload laid out the way the compiler lays one out: the kernel name first,
/// then `key=value` entries, each NUL-terminated, ending with the last entry's
/// NUL. An empty `abi` leaves the key out, which is what a producer that
/// predates the version would have written.
std::string
make_payload(const std::string &name, const std::string &abi,
             const std::vector<std::pair<const char *, std::string>> &entries) {
  std::string payload = name;
  payload.push_back('\0');
  if (!abi.empty()) {
    payload += "abi=";
    payload += abi;
    payload.push_back('\0');
  }
  for (const auto &entry : entries) {
    payload += entry.first;
    payload += entry.second;
    payload.push_back('\0');
  }
  return payload;
}

void the_version_key_is_read() {
  // Pinned literally, because this is the number the compiler stamps and every
  // dispatch library checks. The Rust half of this test asserts the compiler
  // writes exactly `abi=1` into a real payload, so the two ends hold each
  // other.
  check(VX_PAYLOAD_ABI == 1, "the payload version this header defines is 1");

  const std::string with_version = make_payload(
      "vx_npu_kernel_0", std::to_string(VX_PAYLOAD_ABI), {{"kind=", "matmul"}});
  check(vx_payload_abi(with_version.data(), with_version.size()) ==
            VX_PAYLOAD_ABI,
        "a payload carrying this header's version reads back as that version");
  check(vx_payload_field(with_version.data(), with_version.size(), "abi=") !=
            nullptr,
        "and the entry is readable as text too");

  const std::string without_version =
      make_payload("vx_npu_kernel_0", "", {{"kind=", "matmul"}});
  check(vx_payload_abi(without_version.data(), without_version.size()) == -1,
        "a producer that predates the key reports absence, not a version");
  check(vx_payload_field(without_version.data(), without_version.size(),
                         "kind=") != nullptr,
        "and its entries are still readable");

  const std::string empty_version =
      make_payload("vx_npu_kernel_0", "", {{"abi=", ""}});
  check(vx_payload_abi(empty_version.data(), empty_version.size()) == -1,
        "an empty version is not version 0");

  const std::string not_a_number =
      make_payload("vx_npu_kernel_0", "1x", {{"kind=", "matmul"}});
  check(vx_payload_abi(not_a_number.data(), not_a_number.size()) == -1,
        "a version that is not a decimal number is refused");

  const std::string huge =
      make_payload("vx_npu_kernel_0", "99999999999999999999", {});
  check(vx_payload_abi(huge.data(), huge.size()) == -1,
        "a version that overflows is refused rather than wrapped");

  check(vx_payload_abi(nullptr, 0) == -1, "no payload, no version");
}

void entries_still_walk_with_one_more() {
  const std::string payload =
      make_payload("vx_npu_kernel_7", "1",
                   {{"kind=", "matmul"},
                    {"topo=", "500"},
                    {"image=", ".version 7.6\n.target sm_80\n"}});

  const char *kind = vx_payload_field(payload.data(), payload.size(), "kind=");
  check(kind != nullptr && strcmp(kind, "matmul") == 0,
        "a key after the version is still found");
  const char *topo = vx_payload_field(payload.data(), payload.size(), "topo=");
  check(topo != nullptr && strcmp(topo, "500") == 0,
        "and so is one after two others");

  // The name is still the first thing a consumer that ignores all of this
  // reads, which is what makes the whole scheme backward-compatible.
  check(strcmp(payload.c_str(), "vx_npu_kernel_7") == 0,
        "the payload still reads as the kernel name as a C string");

  check(vx_payload_field(payload.data(), payload.size(), "roles=") == nullptr,
        "a key that is not there reports absence");
  check(vx_payload_field(payload.data(), payload.size(), nullptr) == nullptr,
        "no key, no answer");
  check(vx_payload_field(nullptr, 0, "kind=") == nullptr, "no payload");

  // The walk refuses rather than reading past the blob. Truncating the payload
  // so its last entry has no terminator means a lookup that has to walk
  // *through* that entry finds nothing -- it cannot tell absence from
  // truncation, which is the documented answer. A key that appears before it is
  // still found, because the walk stops at the first match and never reaches
  // the damaged tail.
  std::string truncated = payload;
  truncated.resize(truncated.size() - 1);
  check(vx_payload_field(truncated.data(), truncated.size(), "image=") ==
            nullptr,
        "an unterminated entry stops the walk instead of being read past");
  check(vx_payload_field(truncated.data(), truncated.size(), "kind=") !=
            nullptr,
        "a key before the damage is still found");
}

void base64_round_trips() {
  struct Vector {
    const char *encoded;
    const char *decoded;
  };
  // RFC 4648 section 10, which pins the alphabet and the padding rules against
  // something outside this repository.
  const Vector vectors[] = {
      {"", ""},
      {"Zg==", "f"},
      {"Zm8=", "fo"},
      {"Zm9v", "foo"},
      {"Zm9vYg==", "foob"},
      {"Zm9vYmE=", "fooba"},
      {"Zm9vYmFy", "foobar"},
  };
  for (const Vector &v : vectors) {
    unsigned char out[16] = {};
    int64_t n =
        vx_base64_decode(v.encoded, strlen(v.encoded), out, sizeof(out));
    check(n == (int64_t)strlen(v.decoded),
          "a published vector decodes to its own length");
    if (n == (int64_t)strlen(v.decoded))
      check(memcmp(out, v.decoded, (size_t)n) == 0, "and to its own bytes");
  }

  // Our own encoder against the decoder, for lengths that exercise all three
  // padding cases and then some.
  for (size_t len = 0; len <= 20; ++len) {
    std::vector<unsigned char> original(len);
    for (size_t i = 0; i < len; ++i)
      original[i] = (unsigned char)(i * 37 + 5);
    const std::string encoded = b64(original.data(), original.size());
    std::vector<unsigned char> decoded(len + 1);
    int64_t n = vx_base64_decode(encoded.c_str(), encoded.size(),
                                 decoded.data(), decoded.size());
    check(n == (int64_t)len, "the round trip returns the original length");
    if (n == (int64_t)len)
      check(memcmp(decoded.data(), original.data(), len) == 0,
            "and the original bytes");
  }
}

void a_binary_image_survives_the_payload() {
  // The reason `imagebin=` exists. This is the first word of an SPIR-V module,
  // little-endian: three NUL bytes in the first four, so a NUL-terminated entry
  // could carry none of it -- and would drop every entry after it too.
  std::vector<unsigned char> image = {0x03, 0x02, 0x23, 0x07};
  for (size_t i = 0; i < 60; ++i)
    image.push_back((unsigned char)(i == 30 ? 0 : i + 1));

  const std::string payload = make_payload(
      "vx_npu_kernel_3", "1",
      {{"kind=", "matmul"}, {"imagebin=", b64(image.data(), image.size())}});

  check(vx_payload_field(payload.data(), payload.size(), "image=") == nullptr,
        "a binary image is not in the text entry");

  std::vector<unsigned char> decoded(image.size() + 8);
  int64_t n = vx_payload_image_bin(payload.data(), payload.size(),
                                   decoded.data(), decoded.size());
  check(n == (int64_t)image.size(), "the image decodes to its own length");
  if (n == (int64_t)image.size())
    check(memcmp(decoded.data(), image.data(), image.size()) == 0,
          "and to its own bytes, NULs included");
  check(decoded[0] == 0x03 && decoded[1] == 0x02 && decoded[2] == 0x23 &&
            decoded[3] == 0x07,
        "the file magic reads back in the right order");

  // A payload whose image is text has no binary image, and asking for one is
  // absence rather than an error.
  const std::string text_image =
      make_payload("vx_npu_kernel_3", "1", {{"image=", ".version 7.6"}});
  check(vx_payload_image_bin(text_image.data(), text_image.size(),
                             decoded.data(), decoded.size()) == -1,
        "a text image carries no binary image");
}

void refusals() {
  unsigned char out[64] = {};

  // Malformed input: -1.
  struct Bad {
    const char *encoded;
    const char *what;
  };
  const Bad bad[] = {
      {"TWF", "a length that is not a multiple of four"},
      {"TW=u", "a data character after padding"},
      {"T===", "padding before the last two slots"},
      {"====", "padding with nothing to pad"},
      {"TWFu=", "a trailing group of one"},
      {"TW Fu", "a space the producer never writes"},
      {"TWFu\n", "a newline the producer never writes"},
      {"TWF-", "a character outside the alphabet"},
      {"TWF*", "a character outside the alphabet"},
  };
  for (const Bad &b : bad) {
    check(vx_base64_decode(b.encoded, strlen(b.encoded), out, sizeof(out)) ==
              -1,
          b.what);
  }

  // Right input, buffer too small: -2 rather than a truncated decode.
  int64_t too_small = vx_base64_decode("Zm9vYmFy", 8, out, 2);
  check(too_small == -2,
        "a buffer that cannot hold the whole image is a different failure");

  check(vx_base64_decode(nullptr, 0, out, sizeof(out)) == -1, "no input");
  check(vx_base64_decode("Zm9v", 4, nullptr, 0) == -1, "no output buffer");
  check(vx_base64_decode("", 0, out, sizeof(out)) == 0,
        "an empty entry decodes to nothing");
}

} // namespace

int main() {
  the_version_key_is_read();
  entries_still_walk_with_one_more();
  base64_round_trips();
  a_binary_image_survives_the_payload();
  refusals();

  if (failures != 0) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all dispatch payload checks passed\n");
  return 0;
}
