#!/usr/bin/env python3
"""Generate aggregate Change-1 evidence without exporting patient/image identities."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.private_split_audit import read_rows,write_json


def counts(rows,version):
    histogram=Counter((r.get('split','unknown'),r.get('organ','unknown'),r['type'],r.get('gen','unknown')) for r in rows)
    return [dict(version=version,split=a,organ=b,type=c,generator=d,rows=n) for (a,b,c,d),n in sorted(histogram.items())]


def coverage(rows,version):
    result=[]
    for split in ('train','select','test'):
        subset=[r for r in rows if r['split']==split]
        for organ in sorted({r['organ'] for r in subset}):
            selected=[r for r in subset if r['organ']==organ]
            result.append(dict(version=version,split=split,organ=organ,rows=len(selected),**{field:sum(bool(r.get(field)) for r in selected) for field in ('patient_id','volume_id','study_id','video_id','source_id','input_sha256','materialized_sha256')}))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    args=parser.parse_args()
    run=args.run
    paths=json.loads((run/'pipeline_paths.json').read_text())
    raw=json.loads((run/'private_identity_r2/aggregate_audit.json').read_text())
    stage=json.loads((run/'reports/training_exposure.json').read_text())
    if not (run/'pipeline.done').is_file():
        raise SystemExit('No successful pipeline completion marker; refusing a success report')
    if not (run/'final_verification.done').is_file():
        raise SystemExit('Independent final byte/pixel/provenance check has not passed')
    materialized=json.loads((run/'reports/materialized_provenance.json').read_text())
    if any(scope['status']!='PASS' for scope in materialized.values()):
        raise SystemExit('Independent materialized provenance failed')
    folders={name:Path(paths[name]) for name in ('mfclean','extension')}
    current={name:list(read_rows(folder/'manifest.jsonl')) for name,folder in folders.items()}
    stats={name:json.loads((folder/'stats.json').read_text()) for name,folder in folders.items()}
    old=[r for file in ('train_all','select_core','test_core') for r in read_rows(args.data_root/'mfclean'/(file+'.jsonl'))]
    rawrecords={r['path']:r for r in read_rows(run/'private_identity_r2/identity_records_normalized.jsonl')}
    source_gaps=Counter(r['corpus'] for r in rawrecords.values() if r['type']!='real' and (not r.get('source') or r['source'] not in rawrecords))
    old_by_image={r['image']:r for r in old}
    twin_semantics=Counter()
    for row in old:
        if row['type']=='real':
            continue
        target=old_by_image.get(row.get('twin'))
        source=rawrecords[row['orig']].get('source')
        if target is None:
            twin_semantics['missing_or_dangling']+=1
        elif target['orig']==source:
            twin_semantics['exact_mapping_source_path']+=1
        elif rawrecords.get(source,{}).get('pixel_sha256')==rawrecords.get(target['orig'],{}).get('pixel_sha256'):
            twin_semantics['different_path_identical_decoded_source']+=1
        else:
            twin_semantics['different_path_different_decoded_source']+=1
    old_enriched=[]
    for r in old:
        record=rawrecords[r['orig']]
        old_enriched.append(dict(r,patient_id=record.get('patient_id'),volume_id=record.get('volume_id'),source_id=record.get('source')))
    extension_before=[dict(r,split='historical_training') for r in rawrecords.values() if r['corpus']=='extension']
    eval_before=[]
    for r in read_rows(args.data_root/'circe_eval_split_v2.jsonl'):
        record=rawrecords[r['image']]
        eval_before.append(dict(record,split='historical_evaluation'))
    tables=counts(old,'mfclean_historical')+counts(current['mfclean'],'mfclean_change1_v2')+counts(extension_before,'extension_historical')+counts(eval_before,'extension_historical')+counts(current['extension'],'extension_split_v2')
    public=REPO/'docs/evidence/change1_20261101'
    public.mkdir(parents=True,exist_ok=True)
    with (public/'before_after_counts.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['version','split','organ','type','generator','rows'],lineterminator='\n');writer.writeheader();writer.writerows(tables)
    write_json(public/'before_after_counts.json',tables)
    identity_coverage=coverage(old_enriched,'mfclean_historical_reconstructed')+coverage(current['mfclean'],'mfclean_change1_v2')+coverage(current['extension'],'extension_split_v2')
    write_json(public/'identity_coverage.json',identity_coverage)
    write_json(public/'training_exposure.json',stage)
    write_json(public/'materialized_provenance.json',materialized)
    write_json(public/'raw_identity_audit.json',raw)
    write_json(public/'source_and_twin_coverage.json',{'unavailable_authentic_sources_by_corpus':dict(source_gaps),'historical_mfclean_twin_semantics':dict(twin_semantics)})
    mapping=json.loads((args.data_root/'medforge90k_full/dataset_mapping.json').read_text())['items']
    repeated=sum(n-1 for n in Counter(r['image_path'] for r in mapping).values())
    initial=json.loads((run/'logs/historical_declared_split.log').read_text())
    preserved={}
    for row in initial['files'].values():
        target=args.data_root/'mfclean'/row['name']
        preserved[row['name']]=hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
    write_json(public/'preservation_checks.json',{'historical_three_manifest_bytes_unchanged':preserved,'historical_mapping_repeated_image_paths':repeated,
               'writes':'Only new versioned data/run directories and isolated checkout; append-only CPU scheduler entries added. No checkpoint or historical image/manifest was rewritten.'})
    for label in folders:
        write_json(public/(label+'-split-lock.json'),json.loads((run/(label+'-split-lock.json')).read_text()))
    write_json(public/'environment.json',json.loads((run/'environment.json').read_text()))
    write_json(public/'rebuild_summary.json',stats)
    commands=list(read_rows(run/'commands.jsonl'))
    write_json(public/'command_status.json',[{'label':r['label'],'returncode':r['returncode'],'seconds':r['seconds']} for r in commands])
    stage_failures=[r for r in stage['files'] if r.get('cf_forbidden_implants') or r.get('cm_forbidden_editors')]
    raw_cross=raw['historical_mfclean']['cross_split_components']
    lines=['# SPLIT_AUDIT_REPORT — Change 1, private-data execution','',
           '**Status: partially resolved.** Rebuilt prospective splits/pools are separated for the reconstructed identities. Unknown patient/study/video identities and historical test inspection remain unresolved. No training, checkpoint selection, alpha tuning or new model-performance evaluation was run.','',
           '## Code, resources and checks','',
           'Started from the requested fork branch `fix/split-protocol` at `fd894354bebc50595a6d7206058eea84a61997d0`, in an isolated checkout. CPU work ran inside the existing PBS workq allocation `43937.mgmt01` on `gpu01` through its append-only task scheduler, with CUDA disabled, one BLAS thread and four worker processes. No running user job was stopped. Two initial worker setup attempts failed before tests because the compute node lacked Git; corrected wrappers read the already-verified checkout HEAD.','',
           'Base checks: 15 regression tests passed; the final suite passed 21 tests. Scorer/evaluator self-tests passed; the scorer explicitly skipped the absent private legacy-resolver cache. The actual historical `v3data_b` self-tests passed. Both final actual-output pool self-tests passed. Command statuses and private logs retain earlier failures and interrupted attempts.','',
           '## Original mapping and raw evidence','',
           f"The audit rehashed and decoded **{raw['raw_files_rehashed_decoded']:,}** actual files, including raw source/forgery files and historical MF-Clean materializations. Unreadable inputs: **{raw['raw_errors']}**. The mapping contains {raw['mapping_items']:,} records; missing mapping files by type: `{raw['missing_mapping_images_by_type']}`; repeated mapping paths: {repeated}.",'',
           f"Verified near-duplicate links: **{raw['near_duplicates']['full_rgb_verified_links']}**; rejected candidates: **{raw['near_duplicates']['rejected']}**. Candidate rule and full-pixel acceptance thresholds are in raw_identity_audit.json. These are conservative split-safety links, not invented patient identities. Existing declared MF-Clean partitions contain **{raw_cross}** cross-partition reconstructed components. The metadata-only PASS did not establish this raw-file property. The complete raw census was checkpointed; one own audit subprocess was intentionally stopped to resume the same verification rule with vectorized candidate pruning. The first source-incomplete orchestration attempt was stopped before freeze. These are recorded interrupted attempts, not passed data releases.",'',
           'Known filename-derived original-organ patient identities are retained where available. KiTS case keys are recorded as volume/case IDs, not asserted patient IDs. Study IDs, Kvasir/WCE patient/video linkage, and Kaggle kidney patient IDs are unavailable. Zero overlap of declared/reconstructed identities cannot certify those missing identities.','',
           f"Unavailable authentic-source files/links by corpus: `{dict(source_gaps)}`. The original mapping can identify an unavailable source, but that does not provide pixels for construction. Historical twins by exact mapping path versus identical decoded aliases versus different decoded sources: `{dict(twin_semantics)}`. This distinguishes a substituted filename from an actual pixel mismatch.",'',
           '## Rebuild and population changes','',
           '| Scope | Preserved historical rows | Added exact-source real images | Quarantined rows | Historical rows moved to a different split |','|---|---:|---:|---:|---:|']
    for name,s in stats.items():
        lines.append(f"| {name} | {s['historical_rows_preserved']} | {s['added_exact_source_reals']} | {s['quarantined_rows']} | {s['historical_rows_moved_between_splits']} |")
    lines+=['','Every historical MF-Clean example is preserved. Connected source, known patient/case, exact-content, decoded-content and verified-near-duplicate components are merged; conservative historical group unions are retained. Newly rendered identical outputs are also joined before assigning partitions. The canonical existing group and its deterministic hash assignment are retained where possible; actual component merges can move all linked examples together. No accuracy result informed grouping or exclusions.','',
            'Missing and substituted authentic twins are replaced by materialized **exact mapping sources**. The final attempt reuses earlier own-run original-organ renderings only after matching input/source/mask hashes, unchanged renderer code, and output-byte hashes; historical datasets are not modified. Added source images and any reassignment change the evaluation population, so this is protocol **change1-v2r2-20261101**. Every new row is explicitly labelled historically exposed/exploratory. A new directory or split assignment does not create an untouched test set.','',
            'Kidney/colon were rebuilt separately as a split-only version. Their historical 800-image evaluation remains training-exposed; the exact overlap counts are in raw_identity_audit.json. Unresolved sources or missing localization provenance are quarantined with private row-level reasons. All 1,000 kidney injection sources were linked using the generator’s documented space-to-underscore filename conversion and independently rehashed/decoded. The initial incomplete lookup and intermediate build were preserved and not frozen. Format/source/editor confounds and clinical label validity are not repaired by this task. MF-Clean still covers the original organs and does not substitute for extension validation.','',
            f"The source-complete extension r2 pool build failed because full-frame lesion boxes left no surrounding ring for the intensity calculation. That failed attempt is retained. Final extension version `change1-extension-v2r3-20261101` clips **{stats['extension'].get('box_repairs',0)}** declared 1024-grid boxes to inclusive [0,1023], preserves their original values, and retains every image. If a ring is unavailable, the evidence label uses measured source intensity change; zero/unavailable change is explicitly unknown (`none`). This is target construction, not clinical annotation validation. Exact source-copy checks passed 300/300, replacing an inapplicable JPEG-reconstruction test.",'',
            'The complete before/after split × organ × type × editor counts are supplied as CSV and JSON. Identity coverage is supplied by split and organ. Patient identifiers, raw image paths, component membership and quarantine details remain in the private versioned directories.','',
            '## Training stages, locks and historical exposure','',
            f"Audited {len(stage['files'])} candidate historical/new training-stage files. Historical CF/CM exclusion violations occur in {len(stage_failures)} file inventory (not a claim that every candidate file trained a checkpoint). Consumer-provenance checks confirm **4,678 implant references / 2,339 unique implants** in the old CF replay file. Full stage inventories distinguish reference counts, unique images and source-component exposure.",'',
            'New SFT/RL pools, unique continuation controls, pair-exposure continuation controls and SFT-pool replay eligibility inventories were generated. The latter are eligibility lists, not model-mined failures. The label-construction threshold was fixed at the recorded 1.0 value; the explicit override no longer computes an implant-based tuning curve. SFT versus RL groups are disjoint under the reconstructed grouping. CF excludes implants and CM excludes the declared held-out editors in every new applicable inventory. Each continuation control intentionally uses eligible RL identities and represents an alternative experiment, not another sequential SFT-pool stage.','',
            'Manifest freeze and immediate verification ran only after successful raw provenance, pool self-tests and new-stage exposure checks. Both scope-specific split locks and a SHA-256 pool lock are in the private run directory; aggregate split locks are copied here. Image and input hashes are retained in private manifests/census records. No old checkpoint is certified for the new protocol; original training/replay exposure and prior test inspection cannot be undone.','',
            'An additional independent post-freeze check re-read every final image, verified SHA-256 and decoded-pixel identities, reconstructed raw identity-component membership and source hashes, and required exact-source twin links in the same partition/group. Both scopes passed with zero cross-partition or cross-group decoded-content/raw-identity components, zero changed image hashes, and zero missing/substituted twins. This does not fill unknown patient, study or video fields. Its aggregate evidence is in materialized_provenance.json.','',
            '## Artifacts and next step','',
            'Private artifact locations for this execution:','',f"- Isolated code: `{REPO}`",f"- MF-Clean v2: `{folders['mfclean']}`",f"- Kidney/colon split-only v2: `{folders['extension']}`",f"- Logs, complete identities, duplicate decisions and locks: `{run}`",'',
            'Publishable aggregate evidence is alongside this report. Full command stdout/stderr, source membership, exact paths and image hashes remain private. `code.diff` records the changes against the requested input commit. Historical manifest bytes were checked against the initial audit; checkpoints and historical image data were not written.','',
            'Next: train a new original-organ CF baseline from the base pretrained model, using only the newly frozen SFT-CF pool, inside an allocated GPU task. The saved `next_baseline.sh` verifies the split and pool locks first; it was **not executed**. Checkpoint/config candidates and a validation-only selection policy must be fixed before comparing candidates. Evaluation on these already-inspected test identities remains exploratory. A confirmatory claim requires independent identities and, for patient/video independence, the missing source linkage information.']
    report='\n'.join(lines)+'\n'
    (REPO/'SPLIT_AUDIT_REPORT.md').write_text(report)
    (run/'SPLIT_AUDIT_REPORT.md').write_text(report)
    print(json.dumps(dict(status='partially_resolved',raw_files=raw['raw_files_rehashed_decoded'],raw_cross_split_components=raw_cross,new_stage_failures=stage['new_stage_failures'],aggregate_directory=str(public)),indent=2))


if __name__=='__main__':
    main()
