#!/usr/bin/env python3
"""CPU-only checks of token mapping and full-capture acceptance boundaries."""
import unittest

import numpy as np

from compare_model import aggregate, compare, layer_coverage, passed, quantizer_coverage, ranking


def run_manifest():
    run = {"workload": "teacher", "prefill": "batched", "activation_quantization": True,
           "checkpoints": [], "actquant_callbacks_count": 3960}
    for layer in range(30):
        for stage in ("qkv", "attn_out", "ffn_up_gate", "ffn_down"):
            prefix = [26] if layer == 29 and stage in ("ffn_up_gate", "ffn_down") else list(range(27))
            for positions in [prefix, *[[i] for i in range(27, 59)]]:
                run["checkpoints"].append({"kind": "actquant", "layer": layer,
                                           "node": f"actq_{stage}_out-{layer}", "input_positions": positions})
    return run


class ModelContract(unittest.TestCase):
    def test_complete_coverage_accepts_last_layer_cropping(self):
        self.assertTrue(quantizer_coverage(run_manifest(), 33))

    def test_missing_stage_is_not_accepted(self):
        run = run_manifest()
        run["checkpoints"].pop()
        self.assertFalse(quantizer_coverage(run, 33))

    def test_attention_cannot_be_cropped_like_final_ffn(self):
        run = run_manifest()
        for record in run["checkpoints"]:
            if record["layer"] == 29 and record["node"] == "actq_qkv_out-29" and len(record["input_positions"]) > 1:
                record["input_positions"] = [26]
        self.assertFalse(quantizer_coverage(run, 33))

    def test_off_path_does_not_claim_quantizer_coverage(self):
        run = run_manifest()
        run["activation_quantization"] = False
        self.assertIsNone(quantizer_coverage(run, 33))

    def test_complete_layers_accept_final_selected_rows(self):
        run = run_manifest()
        for layer in range(30):
            run["checkpoints"].append({"kind": "l_out", "layer": layer,
                                       "input_positions": list(range(26 if layer == 29 else 0, 59))})
        self.assertTrue(layer_coverage(run))
        run["checkpoints"][-1]["input_positions"].remove(42)
        self.assertFalse(layer_coverage(run))

    def test_nonfinite_is_a_failure_even_with_zero_reference(self):
        metrics, bad = compare(np.array([0, np.nan], np.float32), np.zeros(2, np.float32))
        self.assertEqual(metrics["nonfinite_actual"], 1)
        self.assertEqual(metrics["frozen_tolerance_failures"], 1)
        self.assertFalse(passed(metrics))
        self.assertIsNone(aggregate([metrics])["relative_L2"])
        self.assertEqual(bad.tolist(), [False, True])

    def test_tie_policy_uses_smallest_token_id(self):
        x = np.zeros(128256, np.float32)
        x[83699] = x[14515] = 1.5
        result = ranking(x)
        self.assertEqual(result["greedy_token_id"], 14515)
        self.assertEqual(result["top1_top2_margin"], 0)
        self.assertEqual(result["max_logit_ties"], 2)


if __name__ == "__main__":
    unittest.main()
