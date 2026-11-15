#pragma once
#ifdef __cplusplus
extern "C" {
#endif
// verify creates fresh output_root; benchmark reuses it and requires its proof.
// fixture_root is actual graph shadow root: weights/layer-0 and step-S/layer-0.
// shader_root contains the iPhoneOS metallibs named in SHADERS.md.
// Returns0 local research protocol success,1 numerical/parity failure,3 error.
int bitnet_run_iphone_operator(const char *fixture_root, const char *shader_root,
                              const char *output_root, const char *mode);
#ifdef __cplusplus
}
#endif
