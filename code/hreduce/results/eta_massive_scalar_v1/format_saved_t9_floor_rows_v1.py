"""Format stored attribution strings as TSV; no numeric conversion or calculation."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent/'environment_continuation/separate_images_v1'
SOURCE = ROOT/'saved_t9_floor_attribution.json'
SOURCE_SHA = '5d7c7b31f5b708bc105d7bd9c8c3c7225e2f85aa16c42895f4dad151aae772fd'


def main():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('Changed stored attribution')
    data = json.loads(raw)
    if data['status'] != 'pass' or data['scientific_evaluations'] != 0:
        raise ValueError('Unexpected source status')
    if sys.argv[1:] == ['--geometry-only']:
        geometry = ROOT/'saved_t9_geometry_rows_v1.tsv'
        with geometry.open('x', newline='') as stream:
            out = csv.writer(stream, delimiter='\t')
            out.writerow(('id','parent_id','z_lower','z_upper','t_lower','t_upper',
                          'z_half_width','t_half_width','z_image','t_image',
                          'necessary_change_run','necessary_change_printed'))
            for row in data['children']:
                out.writerow((row['id'],row['parent_id'],*row['bounds'],*row['half_widths'],
                              *row['images'],row['necessary_change']['run'],
                              row['necessary_change']['printed']))
        return
    if sys.argv[1:]:
        raise ValueError('Unknown formatting option')
    children = ROOT/'saved_t9_child_rows_v1.tsv'
    parents = ROOT/'saved_t9_parent_rows_v1.tsv'
    if children.exists() or parents.exists():
        raise FileExistsError('Formatting views already exist')
    with children.open('x', newline='') as stream:
        out = csv.writer(stream, delimiter='\t')
        out.writerow(('id','parent_id','z_lower','z_upper','t_lower','t_upper',
                      'z_half_width','t_half_width','z_image','t_image',
                      'fallback_width','twice_allowance','minimum','limiting',
                      'necessary_change_run','necessary_change_printed'))
        for row in data['children']:
            out.writerow((row['id'],row['parent_id'],*row['bounds'],*row['half_widths'],
                          *row['images'],row['fallback_width'],row['twice_allowance'],
                          row['minimum'],row['limiting'],row['necessary_change']['run'],
                          row['necessary_change']['printed']))
    with parents.open('x', newline='') as stream:
        out = csv.writer(stream, delimiter='\t')
        out.writerow(('id','z_lower','z_upper','t_lower','t_upper','old_width',
                      'refused_minima','passed_minima','uncapped','contribution',
                      'cap_discount','fallback_children_width','child_ids_ranked'))
        for row in data['parents']:
            out.writerow((row['id'],*row['bounds'],row['old_width'],row['refused_minima'],
                          row['passed_minima'],row['uncapped'],row['contribution'],
                          row['cap_discount'],row['fallback_children_width'],
                          json.dumps(row['child_ids'], separators=(',',':'))))


if __name__ == '__main__':
    main()
