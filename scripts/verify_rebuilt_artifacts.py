#!/usr/bin/env python3
"""Verify rebuilt image bytes, decoded pixels and exact-source twin provenance.

No identities are printed; detailed rows remain in the supplied private output.
"""
import argparse
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.private_split_audit import read_rows,write_json,write_rows
from v3.split_protocol import audit_directory


def check_image(row):
    with open(row['image'],'rb') as stream:
        digest=hashlib.sha256(stream.read()).hexdigest()
    with Image.open(row['image']) as image:
        image.load()
        pixels=np.asarray(image.convert('RGB'))
        decoded=hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
    return row['image'],digest==row['materialized_sha256'],decoded


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mfclean',type=Path,required=True)
    parser.add_argument('--extension',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--identities',type=Path,required=True,help='Completed private raw-file identity census')
    args=parser.parse_args()
    if not 1<=args.workers<=7:
        parser.error('Use 1–7 workers in the eight-CPU allocation')
    report={}
    raw={r['path']:r for r in read_rows(args.identities)}
    for name,folder in [('mfclean',args.mfclean),('extension',args.extension)]:
        declared=audit_directory(folder)
        rows=list(read_rows(folder/'manifest.jsonl'))
        by_image={r['image']:r for r in rows}
        if len(by_image)!=len(rows):
            raise ValueError('Repeated manifest image paths')
        identities=defaultdict(set)
        identity_groups=defaultdict(set)
        identity_types=defaultdict(set)
        raw_identity_partitions=defaultdict(set)
        raw_identity_groups=defaultdict(set)
        provenance_mismatch=0
        for row in rows:
            original=raw.get(row['orig'])
            if original is None or original['sha256']!=row['input_sha256']:
                provenance_mismatch+=1
                continue
            source=raw.get(original.get('source'))
            if source is None or source['sha256']!=row['source_sha256']:
                provenance_mismatch+=1
            for field in ('component','content_md5','pixel_sha256','patient_id','volume_id','study_id','video_id'):
                value=original.get(field)
                if value:
                    raw_identity_partitions[(field,value)].add(row['split'])
                    raw_identity_groups[(field,value)].add(row['group'])
        mismatches=0
        with ProcessPoolExecutor(args.workers) as pool:
            for image,good,pixels in pool.map(check_image,rows,chunksize=16):
                mismatches+=not good
                row=by_image[image]
                identities[pixels].add(row['split'])
                identity_groups[pixels].add(row['group'])
                identity_types[pixels].add(row['type'])
        twins=Counter()
        for row in rows:
            if row['type']=='real':
                continue
            twin=by_image.get(row.get('twin'))
            if not twin:
                twins['missing']+=1
            elif twin['split']!=row['split'] or twin['group']!=row['group']:
                twins['wrong_split_or_group']+=1
            elif twin['input_sha256']!=row['source_sha256']:
                twins['wrong_authentic_source_bytes']+=1
            else:
                twins['exact_source']+=1
        cross_split=sum(len(s)>1 for s in identities.values())
        cross_group=sum(len(s)>1 for s in identity_groups.values())
        raw_overlap=Counter(field for (field,value),splits in raw_identity_partitions.items() if len(splits)>1)
        raw_group_overlap=Counter(field for (field,value),groups in raw_identity_groups.items() if len(groups)>1)
        result=dict(image_rows=len(rows),sha256_mismatches=mismatches,decoded_unique_images=len(identities),
                    decoded_content_cross_split_components=cross_split,decoded_content_cross_group_components=cross_group,
                    decoded_content_mixed_real_fake_labels=sum('real' in types and bool(types-{'real'}) for types in identity_types.values()),
                    raw_identity_cross_partition_components=dict(raw_overlap),raw_identity_cross_group_components=dict(raw_group_overlap),
                    raw_input_or_source_provenance_mismatches=provenance_mismatch,
                    twins=dict(twins),declared_audit=declared['audit'])
        result['status']='PASS' if not mismatches and not cross_split and not cross_group and not raw_overlap and not raw_group_overlap and not provenance_mismatch and set(twins)<= {'exact_source'} else 'FAIL'
        report[name]=result
        print(json.dumps({'scope':name,**{key:result[key] for key in ['status','image_rows','decoded_unique_images','sha256_mismatches','decoded_content_cross_split_components']}}),flush=True)
    write_json(args.out,report)
    if any(r['status']!='PASS' for r in report.values()):
        raise SystemExit(1)


if __name__=='__main__':
    main()
