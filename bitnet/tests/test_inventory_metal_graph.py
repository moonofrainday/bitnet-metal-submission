import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("inventory_metal_graph", Path(__file__).with_name("inventory_metal_graph.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class InventoryTests(unittest.TestCase):
    def test_reused_q_name_requires_op(self):
        self.assertEqual(MODULE.classify("MUL_MAT", "Qcur-0"), "attention_projection")
        self.assertEqual(MODULE.classify("ROPE", "Qcur-0"), "attention_rope")
        self.assertEqual(MODULE.classify("RESHAPE", "Qcur-0"), "metadata_noop")

    def test_ffn_weight_and_norm_boundaries(self):
        self.assertEqual(MODULE.classify("MUL_MAT", "unnamed", "blk.29.ffn_up.weight"), "ffn_projection")
        self.assertEqual(MODULE.classify("MUL", "ffn_sub_norm-29"), "normalization")
        self.assertEqual(MODULE.classify("MUL", "ffn_sub_out-29"), "ffn_elementwise")

    def test_head_is_not_attention_matmul(self):
        self.assertEqual(MODULE.classify("MUL_MAT", "result_output"), "output_head")
        self.assertEqual(MODULE.classify("MUL_MAT", "kqv-0"), "attention_core")
        self.assertEqual(MODULE.classify("MUL_MAT", "unknown"), "other_matmul")

    def test_partial_coverage_and_completion_not_added(self):
        text = """ACTQUANT_Q_PHASE_BEGIN phase=decode decode_step=16 count=1 q_route=stock
ACT_QUANT_ENCODE node=actq_qkv_out-0 kernel=kernel_act_quant_f32
BITNET_MULTILAYER_ENCODE node=Qcur-0 weight=blk.0.attn_q.weight pipeline=kernel_mul_mv_tq2_0_f32 command=0x1
BITNET_MULTILAYER_COMPLETE node=Qcur-0 command=0x1 completed=1
BITNET_MULTILAYER_COMPLETE node=Qcur-0 command=0x1 completed=1
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.log"
            path.write_text(text)
            result = MODULE.inventory(path, MODULE.SOURCE_ROOT / "ggml/include/ggml.h")
        self.assertEqual(result["coverage"]["kind"], "targeted_partial")
        self.assertEqual(result["coverage"]["per_operator_GPU_time"], "unavailable")
        self.assertIn("output_head", result["coverage"]["unobserved_families"])
        self.assertEqual(result["observations_by_origin"]["targeted_encode_events"]["count"], 2)
        self.assertEqual(result["completion_events_not_kernel_durations"][0]["count"], 2)

    def test_debug_and_targeted_origins_stay_separate(self):
        text = """ggml_metal_op_encode: node[   12] - MUL_MAT Qcur-0 (concurrent)
ggml_metal_op_encode: node[   13] - VIEW (noop)
BITNET_MULTILAYER_ENCODE node=Qcur-0 weight=blk.0.attn_q.weight pipeline=kernel_mul_mv_tq2_0_f32
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.log"
            path.write_text(text)
            result = MODULE.inventory(path, MODULE.SOURCE_ROOT / "ggml/include/ggml.h")
        self.assertEqual(result["observations_by_origin"]["graph_debug_node_visits"]["count"], 2)
        self.assertEqual(result["observations_by_origin"]["targeted_encode_events"]["count"], 1)
        self.assertFalse(result["coverage"]["complete_graph_confirmed"])

    def test_repeated_phase_has_distinct_scope(self):
        line = "ACTQUANT_Q_PHASE_BEGIN phase=prefill decode_step=0 count=27\nACT_QUANT_ENCODE node=actq_qkv_out-0\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.log"
            path.write_text(line * 2)
            result = MODULE.inventory(path, MODULE.SOURCE_ROOT / "ggml/include/ggml.h")
        self.assertEqual(len(result["phase_observations"]), 2)

    def test_enum_ids_come_from_source(self):
        ids = MODULE.enum_ids(MODULE.SOURCE_ROOT / "ggml/include/ggml.h")
        self.assertEqual(ids["NONE"], 0)
        self.assertLess(ids["ACT_QUANT"], ids["COUNT"])
        self.assertEqual(len(set(ids.values())), len(ids))

    def test_unsupported_enum_expression_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            header = Path(folder) / "ggml.h"
            header.write_text("enum ggml_op { GGML_OP_NONE = 0, GGML_OP_BAD = GGML_OP_NONE + 4, };")
            with self.assertRaises(ValueError):
                MODULE.enum_ids(header)

    def test_natural_CB_overlapping_intervals_are_not_summed(self):
        data = {"schema": 1, "clock": "CLOCK_UPTIME_RAW", "complete": True, "tasks": [{"mode": "S", "operator_family_cpu": [{"family": "output_head", "encode_calls": 1, "encode_ns": 123}]}], "command_buffers": []}
        for command, start, end in [(1, 10.0, 10.2), (2, 10.1, 10.3), (3, 10.4, 10.5)]:
            data["command_buffers"].append({"command": command, "graph": 1, "step": 16, "slice_start": command, "slice_end": command + 1, "encode_start_ns": 400, "completed": True, "status": 4, "gpu_start_s": start, "gpu_end_s": end})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "diag.json"
            path.write_text(json.dumps(data))
            result = MODULE.natural_command_buffers(path)
        self.assertTrue(result["all_observed_CB_intervals_valid"])
        measured = result["GPU_command_buffer_intervals_by_step"][0]
        self.assertAlmostEqual(measured["CB_interval_union_us"], 400000)
        self.assertAlmostEqual(measured["gaps_outside_all_CB_intervals_us"], 100000)
        self.assertNotIn("queue_latency_us", measured)

    def test_duplicate_and_failed_CB_records_are_explicit(self):
        record = {"command": 1, "graph": 1, "step": 1, "slice_start": 0, "slice_end": 9, "encode_start_ns": 100, "completed": False, "status": 5, "gpu_start_s": 0, "gpu_end_s": None}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "diag.json"
            path.write_text(json.dumps({"schema": 1, "clock": "CLOCK_UPTIME_RAW", "tasks": [], "command_buffers": [record, record]}))
            result = MODULE.natural_command_buffers(path)
        self.assertFalse(result["all_observed_CB_intervals_valid"])
        self.assertEqual(len(result["invalid_records"]), 2)


if __name__ == "__main__":
    unittest.main()
