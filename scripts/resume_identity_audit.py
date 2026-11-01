#!/usr/bin/env python3
"""Complete the raw audit from its fully rehashed/decoded private census checkpoint.

Uses exact necessary dHash/thumbnail filters before the same full-RGB verification.
The existing raw-image census and historical datasets are never rewritten.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.private_split_audit import read_rows,write_rows,write_json,build_graph,audit_manifest,near_verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--historical-hashes',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    records=list(read_rows(args.out/'raw_records.jsonl'))
    thumbs=np.load(args.out/'thumbs32.npy',mmap_mode='r')
    if len(records)!=len(thumbs) or any(r.get('error') for r in records):
        raise ValueError('Incomplete/unreadable census checkpoint; cannot resume')
    items=json.loads((args.data_root/'medforge90k_full/dataset_mapping.json').read_text())['items']
    current={r['path']:r for r in records}
    for record in records:
        if (record.get('patient_id') or '').startswith('kits:'):
            record['volume_id']=record['patient_id'];record['patient_id']=None
    expected_existing=sum(str(args.data_root/'medforge90k_full'/item['image_path']) in current for item in items)
    if expected_existing!=sum(r['corpus']=='original' for r in records):
        raise ValueError('Original mapping/checkpoint coverage mismatch')
    historical=[r for name in ('train_all','select_core','test_core') for r in read_rows(args.data_root/'mfclean'/(name+'.jsonl'))]
    if len(historical)!=sum(r['corpus']=='historical_mfclean' for r in records):
        raise ValueError('Historical materialized census coverage mismatch')
    print(f'Resumed complete actual-file census: {len(records)} rehashed/decoded images',flush=True)
    graph,reasons=build_graph(current)
    paths=list(current)
    hashes=np.array([int(r['dhash'],16) for r in records],dtype=np.uint64)
    root_labels={root:n for n,root in enumerate(sorted({graph.f(p) for p in paths}))}
    roots=np.array([root_labels[graph.f(p)] for p in paths])
    thumb_array=np.asarray(thumbs,dtype=np.int16)
    means=np.floor(thumb_array.mean(axis=(1,2))/3).astype(int)
    lut=np.array([n.bit_count() for n in range(65536)],dtype=np.uint8)
    blocks=defaultdict(list)
    candidates=[]
    dhits=0
    # Identical decoded RGB pixels have identical full-pixel verification behavior;
    # source/content union already connects every such copy. One representative
    # therefore retains every possible cross-component near-duplicate edge.
    representatives=[]
    pixels=set()
    for i,record in enumerate(records):
        if record['pixel_sha256'] not in pixels:
            pixels.add(record['pixel_sha256']);representatives.append(i)
    print(f'Near search over {len(representatives)} distinct decoded images',flush=True)
    for n,i in enumerate(representatives,1):
        record=records[i];value=int(hashes[i]);hits=set();bucket=int(means[i])
        for shift,width in ((0,13),(13,13),(26,13),(39,13),(52,12)):
            band=(value>>shift)&((1<<width)-1)
            # Thumbnail MAE<=3 implies global-mean difference<=3. The extra bin
            # on either side makes the pruning conservative at floating boundaries.
            for meanbin in range(bucket-2,bucket+3):
                hits.update(blocks[(record['organ'],shift,band,meanbin)])
            blocks[(record['organ'],shift,band,bucket)].append(i)
        if hits:
            indices=np.fromiter(hits,dtype=np.int32)
            indices=indices[roots[indices]!=roots[i]]
            xor=np.bitwise_xor(hashes[indices],hashes[i])
            distance=lut[xor.view(np.uint16).reshape(-1,4)].sum(axis=1)
            indices=indices[distance<=4]
            dhits+=len(indices)
            if len(indices):
                distance=np.abs(thumb_array[indices]-thumb_array[i]).mean(axis=(1,2))
                for j in indices[distance<=3]:
                    candidates.append((paths[int(j)],paths[i]))
        if n%10000==0:
            print(f'Vectorized near search {n}/{len(representatives)}; candidates={len(candidates)}',flush=True)
    accepted=rejected=0
    decisions=args.out/'near_duplicate_decisions.jsonl'
    if decisions.exists():
        raise ValueError('Near-decision output already exists; preserve the prior attempt and use a new output name')
    with ProcessPoolExecutor(args.workers) as pool,decisions.open('x') as stream:
        for first,second,match,metrics in pool.map(near_verify,candidates,chunksize=16):
            stream.write(json.dumps(dict(first=first,second=second,accepted=match,**metrics))+'\n')
            if match:
                graph.u(first,second);accepted+=1
            else:
                rejected+=1
    for record in records:
        record['component']='component:'+hashlib.sha256(graph.f(record['path']).encode()).hexdigest()[:24]
    write_rows(args.out/'identity_records.jsonl',records)
    original,_=audit_manifest(historical,graph,current)
    old={}
    with args.historical_hashes.open() as stream:
        for row in csv.DictReader(stream,delimiter='\t'):
            old[row['path']]=row['md5']
    comparison=Counter()
    for path,row in current.items():
        if path in old:
            comparison['compared']+=1;comparison['changed']+=old[path]!=row['content_md5']
        else:
            comparison['not_in_historical_census']+=1
    evaluation={r['image'] for r in read_rows(args.data_root/'circe_eval_split_v2.jsonl')}
    extension_overlap={}
    for name in ('circe_sft_v2.jsonl','circe_sft_v3.jsonl'):
        training={r['images'][0] for r in read_rows(args.data_root/name)}
        groups={graph.f(p) for p in training}
        extension_overlap[name]=dict(unique_images=len(training),exact_eval_paths=len(evaluation&training),eval_components_exposed=sum(graph.f(p) in groups for p in evaluation))
    missing=Counter(item['type'] for item in items if str(args.data_root/'medforge90k_full'/item['image_path']) not in current)
    report=dict(schema='medforge-private-identity-audit-v2',mapping_items=len(items),missing_mapping_images_by_type=dict(missing),
                raw_files_rehashed_decoded=len(records),raw_errors=0,historical_hash_comparison=dict(comparison),graph_link_counts=dict(reasons),
                census_checkpoint_reused=True,census_checkpoint_sha256=hashlib.sha256((args.out/'raw_records.jsonl').read_bytes()).hexdigest(),
                near_duplicates=dict(dhash_cross_component_candidates=dhits,thumbnail_candidates=len(candidates),full_rgb_verified_links=accepted,rejected=rejected,
                                     rule='same organ/dimensions; dHash Hamming<=4; thumbnail MAE<=3; full RGB MAE<=3 and <=5% channel values differ by >10; exact decoded copies represented once'),
                historical_mfclean=original,extension_historical_eval_rows=len(evaluation),extension_legacy_overlap=extension_overlap,
                identity_limitations=dict(patient='Only known original-organ filename conventions; unknown remains null',study='unavailable',video='Kvasir/WCE video/patient linkage unavailable',kidney_case='KiTS volume/case keys, not asserted patient IDs; Kaggle kidney patient IDs unavailable',near_duplicate_meaning='conservative split-safety components, not inferred patient identities'))
    write_json(args.out/'aggregate_audit.json',report)
    print(json.dumps(dict(raw_images=len(records),verified_near_links=accepted,cross_partition_components=original['cross_split_components'])),flush=True)


if __name__=='__main__':
    main()
