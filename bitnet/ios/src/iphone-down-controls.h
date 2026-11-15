#pragma once
#ifdef __cplusplus
extern "C" {
#endif
// verify completes all three numerical paths; returns 1 for measured FP64 failure, 2 for execution error.
int bitnet_run_iphone_down(const char *fixture_root, const char *library_root,
                          const char *output_root, const char *mode);
#ifdef __cplusplus
}
#endif
