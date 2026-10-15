"""Export the locked checkpoint and compare CPU inference on six validation images.

This checks format conversion; it does not tune on or reevaluate the test set.
"""
from pathlib import Path
import hashlib
import json
import os
import platform
import statistics
import time

BASE = Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR'] = str(BASE / 'tool_config')
os.environ['MPLBACKEND'] = 'Agg'


def main():
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    torch.set_num_threads(4)
    checkpoint = BASE / 'day14_runs/baseline_30epochs/weights/best.pt'
    output = BASE / 'day14_runs/onnx_verification'
    output.mkdir(exist_ok=True)
    exported = Path(YOLO(str(checkpoint)).export(
        format='onnx', imgsz=320, batch=1, device='cpu',
        half=False, dynamic=False, simplify=False, opset=17, nms=False))
    onnx.checker.check_model(str(exported))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(exported), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    pt = YOLO(str(checkpoint))
    pt.model.eval().fuse()
    converted = YOLO(str(exported), task='detect')
    image_dir = BASE / 'steel_defect_dataset/images/val'
    samples = [sorted(image_dir.glob(name + '_*.jpg'))[0]
               for name in pt.names.values()]
    records = []
    last_tensor = None
    for picture in samples:
        original = cv2.imread(str(picture))
        assert original is not None, picture
        resized = LetterBox(new_shape=(320, 320), auto=False)(image=original)
        array = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1))[None]
        array = array.astype(np.float32) / 255.0
        tensor = torch.from_numpy(array)
        with torch.inference_mode():
            reference = pt.model(tensor)[0].cpu().numpy()
        actual = session.run(None, {session.get_inputs()[0].name: array})[0]
        difference = np.abs(reference - actual)
        record = {'image': picture.name, 'raw_output_shape': list(actual.shape),
                  'max_absolute_difference': float(difference.max()),
                  'mean_absolute_difference': float(difference.mean()),
                  'raw_allclose_rtol_0_001_atol_0_001': bool(np.allclose(
                      reference, actual, rtol=1e-3, atol=1e-3))}
        predictions = []
        for label, model in [('pytorch', pt), ('onnx', converted)]:
            result = model.predict(str(picture), imgsz=320, rect=False,
                                   conf=0.25, iou=0.7, device='cpu',
                                   save=False, verbose=False)[0]
            result.save(filename=str(output / (label + '_' + picture.name)))
            predictions.append(result.boxes.data.cpu().numpy())
        # Pair detections by class and nearest coordinates, without relying on order.
        first, second = predictions
        remaining = set(range(len(second)))
        matched = len(first) == len(second)
        largest_box_delta = 0.0
        largest_conf_delta = 0.0
        for box in first:
            choices = [(float(np.max(np.abs(box[:4] - second[j, :4]))), j)
                       for j in remaining if int(box[5]) == int(second[j, 5])]
            if not choices:
                matched = False
                continue
            delta, index = min(choices)
            remaining.remove(index)
            conf_delta = float(abs(box[4] - second[index, 4]))
            largest_box_delta = max(largest_box_delta, delta)
            largest_conf_delta = max(largest_conf_delta, conf_delta)
            matched = matched and delta <= 0.01 and conf_delta <= 0.001
        record.update({'pytorch_boxes': len(first), 'onnx_boxes': len(second),
                       'detections_match_box_0_01px_conf_0_001': bool(matched),
                       'max_matched_box_delta_px': largest_box_delta,
                       'max_matched_conf_delta': largest_conf_delta})
        records.append(record)
        last_tensor = tensor
        print('SAMPLE', json.dumps(record), flush=True)

    # Warmed, sequential batch-1 model execution, excluding preprocessing/NMS/I/O.
    torch.set_num_threads(4)
    array = last_tensor.numpy()
    timings = {}
    for name in ['pytorch', 'onnxruntime']:
        measured = []
        with torch.inference_mode():
            for i in range(35):
                start = time.perf_counter()
                if name == 'pytorch':
                    pt.model(last_tensor)
                else:
                    session.run(None, {session.get_inputs()[0].name: array})
                elapsed = (time.perf_counter() - start) * 1000
                if i >= 5:
                    measured.append(elapsed)
        timings[name] = {'median_ms': statistics.median(measured),
                         'p95_ms': float(np.percentile(measured, 95)),
                         'warmup': 5, 'measured_runs': len(measured)}
    passed = all(r['raw_allclose_rtol_0_001_atol_0_001'] and
                 r['detections_match_box_0_01px_conf_0_001'] for r in records)
    report = {'checkpoint': str(checkpoint), 'onnx': str(exported),
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'onnx_sha256': hashlib.sha256(exported.read_bytes()).hexdigest(),
              'onnx_bytes': exported.stat().st_size,
              'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                           'ultralytics': ultralytics.__version__, 'onnx': onnx.__version__,
                           'onnxruntime': ort.__version__},
              'settings': {'opset': 17, 'imgsz': 320, 'batch': 1, 'half': False,
                           'dynamic': False, 'simplify': False, 'nms_in_graph': False,
                           'cpu_threads': 4, 'conf': 0.25, 'nms_iou': 0.7},
              'onnx_input': {'name': session.get_inputs()[0].name,
                             'shape': session.get_inputs()[0].shape},
              'onnx_outputs': [{'name': o.name, 'shape': o.shape} for o in session.get_outputs()],
              'samples': records, 'passed': passed, 'forward_only_cpu_timing': timings,
              'limits': 'Six validation examples only, not a full accuracy or end-to-end latency benchmark. CPU local inference, no embedded deployment.'}
    (output / 'verification_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('VERIFICATION_PASSED', passed, 'TIMINGS', json.dumps(timings), flush=True)
    assert passed, 'Conversion check failed; inspect verification_report.json'


if __name__ == '__main__':
    main()
