#!/usr/bin/env python3
"""Synthetic CPU-only checks; these are not measured GPU results."""
import copy
import struct
import unittest
from analyze_mac_q_encoder_probe import analyze


def fixture(mode="target"):
    args = bytearray(120)
    for off, code, vals in [(0, "3i", (2560,2560,1)), (16,"4Q",(66,660,1689600,1689600)),
                            (48,"3i",(2560,1,1)), (64,"4Q",(4,10240,10240,10240)),
                            (96,"3i2hf",(2560,1,4,1,1,0.0))]:
        struct.pack_into("<"+code,args,off,*vals)
    bind = lambda buf, off, length: {"buffer":buf,"offset":off,"buffer_length":length,"hazard_tracking_mode":1}
    w = {"name":"blk.0.attn_q.weight","type":"tq2_0","ne":[2560,2560,1,1],"nb":[66,660,1689600,1689600],"bytes":1689600,"binding":bind(1,0,1689600)}
    x = {"name":"actq_qkv_out-0","type":"f32","ne":[2560,1,1,1],"nb":[4,10240,10240,10240],"bytes":10240,"binding":bind(2,0,20480)}
    y = copy.deepcopy(x);y["name"]="Qcur-0";y["binding"]=bind(2,10240,20480)
    buffers = {"1":w["binding"],"2":x["binding"],"3":y["binding"],"4":y["binding"],"5":x["binding"]}
    records=[]; cbs=[]
    for i in range(32):
        rec={"step":i+1,"graph":10,"graph_index":100,"command":i+1000,"scope_host_ns":10000+i*1000,
             "node":"Qcur-0","op":"MUL_MAT","weight":w,"input":x,"output":y,"mode":mode,
             "planned_pipeline":"kernel_mul_mv_tq2_0_f32","planned_native_PSO":1,"planned_pipeline_wrapper":2,
             "planned_args_hex":args.hex(),"planned_args_bytes":120,"pipeline_nr0":4,"pipeline_nr1":1,"pipeline_nsg":2,
             "pipeline_execution_width":32,"pipeline_max_threads":1024,"planned_buffers":buffers,"planned_grid":[320,1,1],"planned_threads":[32,2,1],"planned_smem":0,
             "scope_dispatches":int(mode=="target"),"actual_buffers":{},"sample_start_index":2*i,"sample_end_index":2*i+1,
             "encoder_scope_ended":True,"tuple_verified":True,"consumed_graph_nodes_required":1,
             "prefix_encoder":1,"selected_encoder":2,"suffix_encoder":3,
             "raw_gpu_start":2000+i*100,"raw_gpu_end":2050+i*100,"GPU_command_completed":True,"natural_CB_matches":1,"valid":True,"elapsed_us":.1}
        if mode=="target":
            for k in ("native_PSO","args_hex","args_bytes","buffers","grid","threads","smem","pipeline"):
                rec["actual_"+k]=rec["planned_"+k]
        records.append(copy.deepcopy(rec))
        cbs.append({"command":rec["command"],"graph":10,"step":i+1,"slice_start":0,"slice_end":1000,"encode_start_ns":rec["scope_host_ns"]-1,"encode_end_ns":rec["scope_host_ns"]+1,"completed":True,"status":4,"gpu_start_s":1+i*.1,"gpu_end_s":1.01+i*.1})
    return {"complete":True,"dropped_records":0,"command_buffer_drops":0,"pending_callback_handlers":0,"callback_wait_timed_out":False,"command_buffers":cbs,
            "selected_q_encoder_probe":{"enabled":True,"mode":mode,"ready":True,"complete":True,"error":"none","dropped_records":0,
            "extra_command_buffers":0,"extra_commits":0,"extra_CPU_waits":0,"counter_buffer_allocations_in_target_encode":0,"fence_allocations_in_target_encode":0,
            "clock_samples_taken_outside_decode_wall":True,"records_count":32,"cpu_before":1000,"cpu_after":21000,"gpu_before":1000,"gpu_after":11000,"records":records}}


class TestProbe(unittest.TestCase):
    def good(self, d, mode="target", **kwargs):
        return analyze(d,mode,**kwargs)["measurement_integrity_pass"]
    def test_complete_target(self): self.assertTrue(self.good(fixture()))
    def test_empty_separate(self): self.assertTrue(self.good(fixture("empty"),"empty"))
    def test_missing_step(self):
        d=fixture();d["selected_q_encoder_probe"]["records"].pop();self.assertFalse(self.good(d))
    def test_duplicate_CB_join(self):
        d=fixture();d["command_buffers"].append(copy.deepcopy(d["command_buffers"][0]));self.assertFalse(self.good(d))
    def test_wrong_geometry(self):
        d=fixture();d["selected_q_encoder_probe"]["records"][0]["actual_threads"]=[32,1,1];self.assertFalse(self.good(d))
    def test_wrong_abi(self):
        d=fixture();raw=bytearray.fromhex(d["selected_q_encoder_probe"]["records"][0]["planned_args_hex"]);struct.pack_into("<i",raw,0,5120);d["selected_q_encoder_probe"]["records"][0]["planned_args_hex"]=raw.hex();self.assertFalse(self.good(d))
    def test_wrong_binding_bounds(self):
        d=fixture();d["selected_q_encoder_probe"]["records"][0]["input"]["binding"]["buffer_length"]=100;self.assertFalse(self.good(d))
    def test_CB_incomplete(self):
        d=fixture();d["command_buffers"][0]["completed"]=False;self.assertFalse(self.good(d))
    def test_clock_wrong_conversion(self):
        d=fixture();d["selected_q_encoder_probe"]["records"][0]["elapsed_us"]=1;self.assertFalse(self.good(d))
    def test_outside_raw_bracket(self):
        d=fixture();d["selected_q_encoder_probe"]["records"][0]["raw_gpu_start"]=999;self.assertFalse(self.good(d))
    def test_explicit_CB_count_only(self):
        self.assertFalse(self.good(fixture(),expected_cbs=64));self.assertTrue(self.good(fixture(),expected_cbs=32))


if __name__=="__main__": unittest.main()
