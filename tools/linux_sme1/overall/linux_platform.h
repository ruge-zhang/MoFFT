#pragma once
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sched.h>
#include <sstream>
#include <string>
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <sys/utsname.h>
#include <unistd.h>

inline std::string linux_read_token(const char *path) {
  std::ifstream in(path);
  std::string value;
  if (!(in >> value)) return "unavailable";
  return value;
}
inline std::string sysstr(const char *key) {
  struct utsname u{};
  if (uname(&u)) return "unavailable";
  if (!std::strcmp(key, "hw.model"))
    return std::string(u.machine) + "/MIDR=" +
      linux_read_token("/sys/devices/system/cpu/cpu0/regs/identification/midr_el1");
  if (!std::strcmp(key, "kern.osproductversion")) return u.release;
  return "unavailable";
}
// Apple thermal level has no equivalent here. Preserve its unavailable value;
// target admission must use independent Linux node/environment observations.
inline long long sysint(const char *) { return -1; }
inline std::string linux_hostname() {
  char host[256]{};
  return gethostname(host, sizeof(host)-1) ? "unavailable" : host;
}
inline bool linux_capture_admission() {
#ifdef MOFFT_LOGIN_MOCK
  std::cerr << "LOGIN_MOCK_ONLY_NOT_TARGET_NUMERICAL_OR_PERFORMANCE_EVIDENCE\n";
  return true;
#else
  const char *job = std::getenv("CCS_JOB_ID");
  if (!job || !*job || std::strspn(job, "0123456789") != std::strlen(job) ||
      linux_hostname().rfind("cn", 0) != 0) {
    std::cerr << "TARGET_CAPTURE_REQUIRES_COMPUTE_JOB_AND_CN_HOST\n";
    return false;
  }
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) ||
      CPU_COUNT(&allowed) != 1 || !CPU_ISSET(0, &allowed)) {
    std::cerr << "TARGET_CAPTURE_REQUIRES_CPU0_SINGLE_CPU_AFFINITY\n";
    return false;
  }
  const unsigned long needed = (1UL << 23) | (1UL << 25) | (1UL << 29);
  const long sve = prctl(51, 0, 0, 0, 0), sme = prctl(64, 0, 0, 0, 0);
  if (!(getauxval(AT_HWCAP) & (1UL << 22)) ||
      (getauxval(AT_HWCAP2) & needed) != needed ||
      sve < 0 || sme < 0 || (sve & 65535) != 64 || (sme & 65535) != 64) {
    std::cerr << "TARGET_SVE_SME_FP64_OR_VECTOR_LENGTH_UNAVAILABLE\n";
    return false;
  }
  return true;
#endif
}
#ifdef MOFFT_LOGIN_MOCK
#define MOFFT_RESULT_TAG "login-mock"
#define MOFFT_EVIDENCE_CLASS "login_mock_not_target_evidence"
#else
#define MOFFT_RESULT_TAG "mofft"
#define MOFFT_EVIDENCE_CLASS "target_capture_requires_external_acceptance"
#endif
