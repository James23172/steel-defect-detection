"""One-epoch pipeline check on 24 training and 12 validation images.

Not a full experiment and not evidence of useful steel-defect accuracy.
"""
import os
from pathlib import Path
import json
import csv

BASE = Path(__file__).resolve().parent
CONFIG = BASE / 'tool_config'
CONFIG.mkdir(exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(CONFIG)
os.environ['MPLBACKEND'] = 'Agg'


def main():
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model = YOLO(str(BASE / 'yolov8n.pt'))
    model.train(data=str(BASE/'day14_smoke_dataset'/'data.yaml'),
                epochs=1, imgsz=320, batch=4, device='cpu', workers=0,
                seed=42, deterministic=True, pretrained=True,
                project=str(BASE/'day14_runs'), name='smoke_1epoch',
                exist_ok=False, plots=True, verbose=False, save=True)
    run = Path(model.trainer.save_dir)
    best = run/'weights'/'best.pt'
    last = run/'weights'/'last.pt'
    assert best.is_file() and last.is_file()
    with (run/'results.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    reloaded = YOLO(str(best))
    assert list(reloaded.names.values()) == ['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']
    picture = sorted((BASE/'day14_smoke_dataset'/'images'/'val').glob('*.jpg'))[0]
    result = reloaded.predict(source=str(picture), imgsz=320, device='cpu',
                              conf=0.25, save=True, project=str(run), name='reload_prediction',
                              verbose=False)[0]
    report = {'purpose':'One-epoch pipeline check only; not final accuracy evaluation',
              'training_images':24,'validation_images':12,'epochs':1,'device':'cpu',
              'run_directory':str(run),'best_checkpoint':str(best),'last_checkpoint':str(last),
              'training_csv_row':rows[0],'reload_prediction_image':picture.name,
              'reload_prediction_boxes':len(result.boxes),
              'completed_checks':['real training updates','validation','checkpoint saving','checkpoint reload','single-image inference']}
    (BASE/'day14_smoke_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SMOKE_COMPLETE',flush=True)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
