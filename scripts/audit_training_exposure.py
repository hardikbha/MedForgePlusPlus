#!/usr/bin/env python3
"""Audit candidate historical/current training manifests against reconstructed identities.

Presence in a manifest is exposure opportunity, not proof a checkpoint consumed it.
Private output includes paths; publish only the aggregate report.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.private_split_audit import read_rows, write_json
from scripts.rebuild_split_protocol import assign_components, FILES
from v3.build_mfclean import CM_HOLDOUT


def references(row):
    if isinstance(row,dict):
        if isinstance(row.get('image'),str):
            yield row['image']
        for path in row.get('images',[]):
            if isinstance(path,str):
                yield path
        for key in ('fake','real'):
            if isinstance(row.get(key),dict):
                yield from references(row[key])


def regime_of(name):
    lower=name.lower()
    cf='crossforgery' in lower or bool(re.search(r'(?:^|_)cf(?:cm)?(?:_|\.|$)',lower))
    cm='crossmodel' in lower or 'cfcm' in lower or bool(re.search(r'(?:^|_)cm(?:_|\.|$)',lower))
    return cf,cm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('data-root','audit-dir','mfclean-new','extension-new','out'):
        parser.add_argument('--'+option,type=Path,required=True)
    args=parser.parse_args()
    records={r['path']:r for r in read_rows(args.audit_dir/'identity_records.jsonl')}
    historical=[r for filename in FILES.values() for r in read_rows(args.data_root/'mfclean'/filename)]
    groups,_=assign_components(records,historical)
    aliases={row['image']:row['orig'] for row in historical}
    partitions={}
    redirects={}
    for scope,folder in [('mfclean',args.mfclean_new),('extension',args.extension_new)]:
        partitions[scope]={}
        for split,filename in FILES.items():
            rows=list(read_rows(folder/filename))
            partitions[scope][split]={row['group'] for row in rows}
            for row in rows:
                aliases[row['image']]=row['orig']
                # Includes final unions discovered from actual materialized bytes.
                previous=groups[row['orig']]
                if previous in redirects and redirects[previous]!=row['group']:
                    raise ValueError('A reconstructed component was split between different final groups')
                redirects[previous]=row['group']
    def canonical(group):
        seen=set()
        while group in redirects and redirects[group]!=group:
            if group in seen:
                raise ValueError('Cyclic final-group redirection')
            seen.add(group)
            group=redirects[group]
        return group
    groups={path:canonical(group) for path,group in groups.items()}
    md5lookup=defaultdict(list)
    for record in records.values():
        if record.get('content_md5'):
            md5lookup[record['content_md5']].append(record['path'])
    resolved_cache={}
    def resolve(path):
        if path in resolved_cache:
            return resolved_cache[path]
        original=aliases.get(path,path)
        if original not in records and '/MedForge-90K/' in original:
            original=str(args.data_root/'medforge90k_full'/original.split('/MedForge-90K/',1)[1])
        if original not in records:
            # Resolve actual content, never a guessed basename/patient identifier.
            candidate=Path(path)
            if candidate.is_file():
                matches=md5lookup.get(hashlib.md5(candidate.read_bytes()).hexdigest(),[])
                if matches:
                    original=matches[0]
        record=records.get(original)
        value=(original,groups.get(original),record) if record else (None,None,None)
        resolved_cache[path]=value
        return value
    for scope,files in [('legacy_original',{'select':'mfp_v2_val_indomain_eval.jsonl','test':'mfp_v2_test_shared.jsonl'}),
                        ('legacy_extension',{'test':'circe_eval_split_v2.jsonl'})]:
        partitions[scope]={}
        for split,filename in files.items():
            components=set()
            for row in read_rows(args.data_root/filename):
                for path in references(row):
                    _,component,_=resolve(path)
                    if component:
                        components.add(component)
            partitions[scope][split]=components
    candidates=[]
    for path in sorted(args.data_root.glob('*.jsonl')):
        if path.name.startswith(('gen_','gens_')):
            continue
        if any(word in path.name for word in ('train','sft','replay','raft','_rl_')) or path.name.startswith('rl_'):
            candidates.append(('historical/root',path))
    for folder in [args.data_root/'mfclean/v3data',args.data_root/'mfclean/v3data_b']:
        for path in sorted(folder.glob('*.jsonl')):
            if path.name.startswith(('sft_','rl_','contsft_')):
                candidates.append(('historical/'+folder.name,path))
    for scope,folder in [('new/mfclean',args.mfclean_new),('new/extension',args.extension_new)]:
        for path in sorted((folder/'v3data_v2').glob('*.jsonl')):
            if path.name.startswith(('sft_','rl_','contsft_','replay_eligible_')):
                candidates.append((scope,path))
    summary=[]
    new_sets=defaultdict(dict)
    for kind,path in candidates:
        seen=set(); components=set(); types=Counter(); reference_types=Counter(); generators=Counter(); unresolved=set(); repeated=0; references_n=0
        for row in read_rows(path):
            for image in references(row):
                references_n+=1
                original,component,record=resolve(image)
                if record is not None:
                    reference_types[record['type']]+=1
                if image in seen:
                    repeated+=1
                    continue
                seen.add(image)
                if record is None:
                    unresolved.add(image)
                    continue
                components.add(component)
                types[record['type']]+=1
                generators[record['gen']]+=1
        cf,cm=regime_of(path.name)
        regime_evidence='manifest filename only; no claim that a checkpoint consumed this candidate'
        if kind=='historical/root' and path.name=='mfp_replay_implant_focus_sft.jsonl':
            # This misnamed file was actually consumed by the historical CF
            # replay checkpoint; filename inference alone misses the violation.
            # Existing log line 1 identifies all 6,400 SFT records from this file;
            # circe/audit_protocol.py and build_strict_failure_replay.py explicitly
            # identify it as the CF stage repaired by the strict replay recipe.
            log=args.data_root/'mfp_crossforgery_implant_replay_sft_8b.log'
            header=log.open().readline() if log.is_file() else ''
            if path.name not in header:
                raise ValueError('Expected historical CF consumer provenance is unavailable')
            cf=True
            regime_evidence='Confirmed CF replay use: mfp_crossforgery_implant_replay_sft_8b.log line 1; circe/audit_protocol.py; circe/build_strict_failure_replay.py'
        overlaps={scope:{split:len(components&ids) for split,ids in splits.items() if split!='train'} for scope,splits in partitions.items()}
        row=dict(family=kind,file=path.name,reference_rows=references_n,unique_image_paths=len(seen),intentional_or_historical_repeated_references=repeated,
                 unresolved_paths=len(unresolved),groups=len(components),types=dict(types),reference_types=dict(reference_types),generators=dict(generators),
                 regime_evidence=regime_evidence,
                 cf_implant_exclusions_applicable=cf,cf_forbidden_implants=types['edit'] if cf else None,
                 cf_forbidden_implant_reference_rows=reference_types['edit'] if cf else None,
                 cm_editor_exclusions_applicable=cm,cm_forbidden_editors=sum(generators[g] for g in CM_HOLDOUT) if cm else None,
                 validation_test_component_exposure=overlaps)
        if kind.startswith('new/'):
            own=kind.split('/')[-1]
            row['status']='PASS' if not unresolved and not any(overlaps[own].values()) and not row['cf_forbidden_implants'] and not row['cm_forbidden_editors'] else 'FAIL'
            new_sets[kind][path.name]=components
        else:
            row['status']='HISTORICAL_EXPOSURE_RECORDED'
            if kind.startswith('historical/v3data'):
                new_sets[kind][path.name]=components
        summary.append(row)
        print(json.dumps(dict(family=kind,file=path.name,unique_paths=len(seen),unresolved=len(unresolved))),flush=True)
    separation=[]
    for kind,files in new_sets.items():
        for reg in ('id','cf','cm','cfcm'):
            sft=files.get('sft_'+reg+'.jsonl',set());rl=files.get('rl_'+reg+'.jsonl',set())
            separation.append(dict(family=kind,regime=reg,sft_rl_group_overlap=len(sft&rl)))
    report=dict(schema='stage-exposure-audit-v2',scope='Candidate manifests; consuming checkpoints not retrospectively certified',files=summary,
                sft_rl_separation=separation,
                new_stage_failures=sum(r['status']=='FAIL' for r in summary),
                historical_test_exposure_warning='Old checkpoints and public test results remain historically exposed; initialize a new baseline from the base pretrained model.')
    write_json(args.out,report)
    if report['new_stage_failures'] or any(r['sft_rl_group_overlap'] for r in separation if r['family'].startswith('new/')):
        raise SystemExit(1)


if __name__=='__main__':
    main()
