#!/usr/bin/env python
"""W5+W6 픽스 통합 convert 실행기.

W5: TRTBEVPoolv2.forward -> export-only 더미 shim (torch 1.13 트레이서 _Map_base::at 우회).
W6: from_onnx 의 deprecated `builder.build_engine` -> 모던 `builder.build_serialized_network`.
    (동일 onnx가 build_engine에선 segfault, build_serialized_network에선 완전 빌드됨을 실측.)

convert 모듈을 importlib로 임포트(main 미실행)해 내부 helper(HDF5CalibratorBEVDet,
create_calib_input_data)를 그대로 재사용하고, from_onnx 만 교체 후 main() 호출.
"""
import os, sys, importlib.util
import onnx
import tensorrt as trt

MODE = os.environ.get("BEVDET_MODE", "fp32")
CONV = os.path.expanduser("~/capstone-bev/BEVDet/tools/convert_bevdet_to_TRT.py")

# ---- W5: TRTBEVPoolv2.forward shim ----
import mmdet3d.ops.bev_pool_v2.bev_pool as bp
def _export_shim_forward(g, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                         interval_starts, interval_lengths,
                         out_height=128, out_width=128):
    feat = feat.unsqueeze(0); depth = depth.unsqueeze(0)
    out_shape = (depth.shape[0], out_height, out_width, feat.shape[-1])
    return feat.new_zeros(out_shape)
bp.TRTBEVPoolv2.forward = staticmethod(_export_shim_forward)
print(f"[W5 patch] TRTBEVPoolv2.forward -> export shim (MODE={MODE})", flush=True)

# ---- convert 모듈 임포트 (main 미실행: __name__ != '__main__') ----
spec = importlib.util.spec_from_file_location("convert_bevdet_mod", CONV)
mod = importlib.util.module_from_spec(spec)
sys.modules["convert_bevdet_mod"] = mod
spec.loader.exec_module(mod)
print("[import] convert module loaded (main not run)", flush=True)

# ---- W6: from_onnx 교체 (build_serialized_network) ----
def _from_onnx_serialized(onnx_model, output_file_prefix, input_shapes,
                          max_workspace_size=1 << 30, fp16_mode=False,
                          int8_mode=False, int8_param=None, device_id=0,
                          log_level=trt.Logger.INFO, **kwargs):
    os.environ['CUDA_DEVICE'] = str(device_id)
    import pycuda.autoinit  # noqa
    mod.load_tensorrt_plugin()

    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)
    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)
    if not parser.parse(onnx_model.SerializeToString()):
        msgs = ''.join(f'{parser.get_error(i)}\n' for i in range(parser.num_errors))
        raise RuntimeError(f'Failed to parse onnx, {msgs}')

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    profile = builder.create_optimization_profile()
    for name, param in input_shapes.items():
        profile.set_shape(name, param['min_shape'], param['opt_shape'], param['max_shape'])
    config.add_optimization_profile(profile)

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8_mode:
        config.set_flag(trt.BuilderFlag.INT8)
        assert int8_param is not None
        config.int8_calibrator = mod.HDF5CalibratorBEVDet(
            int8_param['calib_file'], input_shapes,
            model_type=int8_param['model_type'], device_id=device_id,
            algorithm=int8_param.get('algorithm',
                                     trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2))
        config.set_calibration_profile(profile)  # explicit-batch INT8 필수

    print(f">>> build_serialized_network START (mode={MODE}, fp16={fp16_mode}, int8={int8_mode})", flush=True)
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, 'Failed to build serialized engine'
    out = output_file_prefix + '.engine'
    with open(out, 'wb') as f:
        f.write(bytes(serialized))
    print(f">>> ENGINE SAVED: {out} ({serialized.nbytes} bytes)", flush=True)
    return serialized

mod.from_onnx = _from_onnx_serialized
print("[W6 patch] from_onnx -> build_serialized_network", flush=True)

# ---- main 실행 ----
FLAGS = {"fp32": [], "fp16": ["--fp16"], "int8": ["--fp16", "--int8"]}[MODE]
sys.argv = ["convert_bevdet_to_TRT.py",
            "configs/bevdet/bevdet-r50.py",
            "work_dirs/capstone/init_r50.pth",
            "work_dirs/capstone/trt",
            "--prefix", "bevdet"] + FLAGS
print("[argv]", " ".join(sys.argv), flush=True)
mod.main()
print(f"===== convert2 MODE={MODE} DONE =====", flush=True)
sys.stdout.flush(); sys.stderr.flush()
# pycuda.autoinit + TRT 객체의 인터프리터-종료 teardown 순서 segfault(무해, 엔진은 이미 저장)
# 회피: 정리 스킵하고 즉시 종료해 exit code를 깨끗이 유지.
os._exit(0)
