"""Prepare the NEU-DET mirror for the day-14 learning experiment.

Preserves raw files. Groups byte-identical images before splitting.
Keeps difficult objects as ordinary labeled objects; this is not VOC's
ignore-difficult evaluation protocol. No test data is used for the smoke run.
"""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
RAW = BASE / 'NEU-DET原始镜像'
OUTPUT = BASE / 'steel_defect_dataset'
SMOKE = BASE / 'day14_smoke_dataset'
NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
SEED = 42


def configuration(folder):
    lines = [f'path: {json.dumps(folder.as_posix(), ensure_ascii=False)}',
             'train: images/train', 'val: images/val', 'test: images/test', 'names:']
    lines += [f'  {index}: {name}' for index, name in enumerate(NAMES)]
    (folder / 'data.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def materialize(record, folder, split):
    image_dir, label_dir = folder / 'images' / split, folder / 'labels' / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record['image'], image_dir / record['image'].name)
    (label_dir / (record['stem'] + '.txt')).write_text(record['label'], encoding='utf-8')


def main():
    if OUTPUT.exists() or SMOKE.exists():
        raise RuntimeError('Prepared data already exists; inspect it before creating another version.')
    records = []
    for image_folder, xml_folder in [('IMAGES', 'ANNOTATIONS'), ('Validation_Images', 'Validation_Annotations')]:
        for image in sorted((RAW / image_folder).glob('*.jpg')):
            annotation = ET.parse(RAW / xml_folder / (image.stem + '.xml')).getroot()
            width = int(annotation.findtext('size/width'))
            height = int(annotation.findtext('size/height'))
            lines, classes, boxes = [], [], []
            for obj in annotation.findall('object'):
                category = NAMES.index(obj.findtext('name'))
                box = obj.find('bndbox')
                x1, y1, x2, y2 = [float(box.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
                assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                normalized = [(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height]
                # Round-trip check every converted box against its original coordinates.
                cx, cy, bw, bh = normalized
                recovered = [(cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height]
                assert all(abs(a-b) < 1e-6 for a,b in zip(recovered, [x1,y1,x2,y2]))
                lines.append(str(category) + ' ' + ' '.join(f'{number:.8f}' for number in normalized))
                classes.append(category)
                boxes.append([category, x1, y1, x2, y2])
            records.append({'stem':image.stem, 'image':image, 'sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                            'primary_class':image.stem.rsplit('_',1)[0], 'classes':classes, 'boxes':boxes,
                            'label':'\n'.join(lines)+'\n'})
    assert len(records) == 1800 and sum(len(r['classes']) for r in records) == 4189
    assert len({r['stem'] for r in records}) == len(records)
    groups = defaultdict(list)
    for record in records:
        groups[record['sha256']].append(record)
    strata = defaultdict(list)
    for digest, group in sorted(groups.items()):
        assert len({r['primary_class'] for r in group}) == 1
        strata[group[0]['primary_class']].append(group)
    splits = {'train':[], 'val':[], 'test':[]}
    rng = random.Random(SEED)
    for category in NAMES:
        buckets = strata[category]
        rng.shuffle(buckets)
        train_end = round(len(buckets)*0.70)
        val_end = train_end + round(len(buckets)*0.15)
        for split, selected in [('train', buckets[:train_end]), ('val', buckets[train_end:val_end]), ('test', buckets[val_end:])]:
            splits[split].extend(record for group in selected for record in group)
    hashes = {name:{r['sha256'] for r in values} for name,values in splits.items()}
    assert not (hashes['train'] & hashes['val'] or hashes['train'] & hashes['test'] or hashes['val'] & hashes['test'])
    OUTPUT.mkdir()
    SMOKE.mkdir()
    manifest = {'seed':SEED, 'classes':NAMES,
                'split_method':'Group byte-identical images, stratify by filename prefix, shuffle sorted groups with a fixed seed; 70/15/15 of groups. Does not guarantee balanced object counts or remove all near duplicates.',
                'coordinate_rule':'Use original xmin,ymin,xmax,ymax as continuous image-boundary coordinates; no +/-1 conversion.',
                'difficult_rule':'Keep all 81 difficult objects as ordinary labels. Evaluation differs from VOC ignore-difficult protocol.',
                'source':'NEU-DET原始镜像/来源记录.json', 'splits':{}}
    for split, values in splits.items():
        for record in values:
            materialize(record, OUTPUT, split)
        manifest['splits'][split] = {'images':len(values), 'objects':sum(len(r['classes']) for r in values),
            'objects_by_class':dict(Counter(NAMES[c] for r in values for c in r['classes'])),
            'files':[{'stem':r['stem'],'sha256':r['sha256']} for r in values]}
    configuration(OUTPUT)
    smoke_manifest = {'purpose':'Pipeline check only, not final accuracy evaluation', 'splits':{}}
    for split, count in [('train',4), ('val',2)]:
        selected = []
        for category in NAMES:
            selected.extend([r for r in splits[split] if r['primary_class']==category][:count])
        for record in selected:
            materialize(record, SMOKE, split)
        smoke_manifest['splits'][split] = [r['stem'] for r in selected]
    configuration(SMOKE)
    (OUTPUT/'split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),encoding='utf-8')
    (SMOKE/'split_manifest.json').write_text(json.dumps(smoke_manifest, ensure_ascii=False, indent=2),encoding='utf-8')
    print('Full split:',{split:len(values) for split,values in splits.items()},flush=True)
    print('Smoke split:',{split:len(values) for split,values in smoke_manifest['splits'].items()},flush=True)
    print('Converted 4189 boxes; coordinate round-trip and cross-split exact-duplicate checks passed.',flush=True)


if __name__ == '__main__':
    main()
