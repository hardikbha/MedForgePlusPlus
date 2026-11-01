#!/usr/bin/env python3
"""Rebuild private version-2 split artifacts from a completed raw identity audit.

Preserves every historical MF-Clean example, adds exact-source authentic twins,
and merges connected identities before assigning any split. New test rows remain
exploratory: neither new paths nor regrouping erase historical inspection.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from v3 import build_mfclean as bm
from v3.split_protocol import validate_splits
from scripts.private_split_audit import read_rows, write_json, write_rows

FILES = {"train": "train_all.jsonl", "select": "select_core.jsonl", "test": "test_core.jsonl"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assign_components(records, historical):
    """Keep conservative historical unions; add independently established links."""
    union = bm.UF()
    prior = defaultdict(list)
    for row in historical:
        prior[row["group"]].append(records[row["orig"]]["component"])
    for group, components in prior.items():
        for component in components:
            union.u(components[0], component)
    previous = defaultdict(set)
    for row in historical:
        previous[union.f(records[row["orig"]]["component"])].add(row["group"])
    fingerprints = defaultdict(set)
    for record in records.values():
        if record.get("sha256"):
            fingerprints[union.f(record["component"])].add(record["sha256"])
    groups = {}
    for path, record in records.items():
        component = union.f(record["component"])
        # Retain a canonical old ID when one exists, preventing gratuitous shuffle.
        if component in previous:
            group = min(previous[component])
        else:
            if not fingerprints[component]:
                raise ValueError("Cannot create a reproducible group without input content hashes")
            stable = "\n".join(sorted(fingerprints[component]))
            group = "g2" + hashlib.sha256(stable.encode()).hexdigest()[:20]
        groups[path] = group
    return groups, {key: sorted(value) for key,value in previous.items() if len(value)>1}


def render(job):
    record, destination, raw_root, extension = job
    source = record["source"]
    if extension:
        relative = Path(record.get("subset", "source_only")) / Path(record["path"]).name
        target = Path(destination) / "img" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record["path"], target)
        result = {"ok": True, "image": str(target), "box": record.get("box") if record["type"]!="real" else None}
    else:
        relative = str(Path(record["path"]).relative_to(raw_root))
        cached = record.get("_reuse")
        if cached:
            target = Path(destination) / "img" / (str(Path(relative).with_suffix(".jpg")))
            if sha256(cached["image"]) != cached["materialized_sha256"]:
                raise ValueError("Prior rendering changed; refuse cache reuse")
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(cached["image"],target)
            return dict(ok=True,image=str(target),box=cached.get("box"),path=record["path"],
                        materialized_sha256=sha256(target),reused_render=True)
        row = {"rel": relative, "organ": record["organ"]}
        if record["type"] != "real":
            row.update(src_rel=str(Path(source).relative_to(raw_root)), mask_rel=str(Path(record["mask"]).relative_to(raw_root)))
        result = bm.work(("real" if record["type"]=="real" else "fake", row, destination))
    if not result["ok"]:
        raise RuntimeError("Rendering failed: " + str(result))
    result["materialized_sha256"] = sha256(result["image"])
    result["path"] = record["path"]
    return result


def extension_boxes(root, workspace):
    boxes = {}
    for subset in ("kidney_removal", "kidney_real_tumor", "kidney_normal", "polyp_removal", "polyp_real", "polyp_normal"):
        folder = root / "naitik_annotated" / subset
        for mapping in sorted(folder.glob("mapping*.json")):
            for row in json.loads(mapping.read_text()):
                path = str(folder / "images" / Path(row["image_path"]).name)
                box = row.get("box")
                if box is not None:
                    if path in boxes and boxes[path] != box:
                        raise ValueError("Conflicting extension boxes; requires adjudication")
                    boxes[path] = box
    for organ, folder in (("colon", workspace/"AAAI/colon/flux_injected"),("kidney",workspace/"AAAI/kidney/flux_injected")):
        for manifest in sorted(folder.glob("manifest*.jsonl")):
            for row in read_rows(manifest):
                path = str((root/"colon_injections/images" if organ=="colon" else folder/"images") / Path(row["images"][0]).name)
                box = row["box"]
                if path in boxes and boxes[path] != box:
                    raise ValueError("Conflicting implantation boxes; requires adjudication")
                boxes[path] = box
    return boxes


def build(records, selected, groups, historical, destination, raw_root, workers, extension=False, reuse=None):
    destination.mkdir(parents=True, exist_ok=False)
    # Exact source means exact source path, not an arbitrary duplicate representative.
    # Adding authentic twins changes the evaluation population and is versioned.
    kept, quarantine = {}, []
    for path in selected:
        row = records[path]
        if row.get("error") or not row.get("source") or row["source"] not in records:
            quarantine.append(dict(path=path,reason="missing_or_unreadable_authentic_source_or_raw_input"))
            continue
        if extension and row["type"]!="real" and not row.get("box"):
            quarantine.append(dict(path=path,reason="missing_localization_box"))
            continue
        kept[path] = row
        if row["type"]!="real":
            source = records[row["source"]]
            if source.get("error") or source["type"]!="real":
                raise ValueError("Authentic source is unreadable or not authentic")
            kept[row["source"]] = source
    if not extension and quarantine:
        raise ValueError("Refusing to drop historical MF-Clean examples; investigate raw provenance failures")
    if reuse:
        previous={r["orig"]:r for r in read_rows(reuse/"manifest.jsonl")}
        for path,row in kept.items():
            cached=previous.get(path)
            if cached and cached["input_sha256"]==row["sha256"] and cached["source_sha256"]==records[row["source"]]["sha256"] and cached.get("mask_sha256")==row.get("mask_sha256"):
                row["_reuse"]=cached
    jobs = [(row,str(destination),raw_root,extension) for row in kept.values()]
    outputs = {}
    with ProcessPoolExecutor(workers) as pool:
        for i,result in enumerate(pool.map(render,jobs,chunksize=16),1):
            outputs[result["path"]] = result
            if i%5000==0:
                print(f"rendered {i}/{len(jobs)} {destination.name}",flush=True)
    # Reconcile actual newly materialized identical bytes before final assignment.
    union = bm.UF()
    by_hash = {}
    for path, result in outputs.items():
        group = groups[path]
        previous = by_hash.setdefault(result["materialized_sha256"],group)
        union.u(group,previous)
    final_groups = {path:union.f(groups[path]) for path in kept}
    manifest = []
    for path,row in kept.items():
        output = outputs[path]
        group = final_groups[path]
        twin = outputs[row["source"]]["image"] if row["type"]!="real" else None
        manifest.append(dict(image=output["image"],type=row["type"],gen=row["gen"],box=output.get("box"),organ=row["organ"],
                             group=group,split=bm.split_of(group),disease=row.get("disease", ""),twin=twin,orig=path,
                             source_id=("extension:" if extension else "medforge90k:")+row["source"],
                             patient_id=row.get("patient_id"),volume_id=row.get("volume_id"),study_id=None,video_id=None,
                             duplicate_cluster="split-component:"+group,content_md5=row["content_md5"],
                             source_sha256=records[row["source"]]["sha256"],input_sha256=row["sha256"],
                             materialized_sha256=output["materialized_sha256"],mask_sha256=row.get("mask_sha256"),
                             protocol_version="change1-v2r2-20261101",evaluation_status="historically_exposed_exploratory",
                             reused_verified_same_input_render=bool(output.get("reused_render"))))
    manifest.sort(key=lambda r:(r["split"],r["organ"],r["group"],r["orig"]))
    partitions={split:[r for r in manifest if r["split"]==split] for split in FILES}
    audit=validate_splits(partitions)
    for split,filename in FILES.items():
        write_rows(destination/filename,partitions[split])
    write_rows(destination/"manifest.jsonl",manifest)
    write_rows(destination/"quarantine.jsonl",quarantine)
    old={row["orig"]:row for row in historical}
    transitions=[]
    for row in manifest:
        if row["orig"] in old:
            before=old[row["orig"]]
            transitions.append(dict(orig=row["orig"],before_split=before["split"],after_split=row["split"],before_group=before["group"],after_group=row["group"]))
    write_rows(destination/"private_transitions.jsonl",transitions)
    statistics={"protocol_version":"change1-v2r2-20261101","population":"extension split-only" if extension else "historical MF-Clean population plus exact-source twins",
                "historical_rows_preserved":len(set(selected)&set(kept)),"added_exact_source_reals":len(kept)-len(set(selected)&set(kept)),
                "quarantined_rows":len(quarantine),"render_failures":0,"split_audit":audit,
                "reused_same_input_renderings":sum(bool(o.get("reused_render")) for o in outputs.values()),
                "historical_rows_moved_between_splits":sum(x['before_split']!=x['after_split'] for x in transitions),
                "evaluation_status":"exploratory; regrouping and a new directory do not erase previous model/developer exposure",
                "extension_limitations":"No source/format/editor/clinical repair; unknown patient/video identities remain unknown" if extension else None}
    write_json(destination/"stats.json",statistics)
    return statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--extension-out",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--reuse-mfclean",type=Path,help="Reuse own prior renderings only with matching input/source/mask and output hashes")
    args=parser.parse_args()
    if not 1<=args.workers<=7:
        parser.error("Use 1–7 workers within the eight-CPU allocation")
    records={r["path"]:r for r in read_rows(args.audit_dir/"identity_records.jsonl")}
    # Phase-1 recorded KiTS case keys under patient_id. They are volume/case
    # identities, not independently verified patient IDs; preserve that distinction.
    normalized=0
    for row in records.values():
        if (row.get("patient_id") or "").startswith("kits:"):
            row["volume_id"]=row["patient_id"]
            row["patient_id"]=None
            normalized+=1
    write_json(args.audit_dir/"identity_field_normalization.json",{"kits_case_keys_retyped_as_volume_id":normalized,"patient_ids_invented":0,"component_membership_changed":False})
    write_rows(args.audit_dir/"identity_records_normalized.jsonl",records.values())
    report=json.loads((args.audit_dir/"aggregate_audit.json").read_text())
    if report["raw_errors"]:
        raise ValueError("Raw identity audit recorded unreadable inputs; resolve before rebuilding")
    historical=[]
    for filename in FILES.values():
        historical.extend(read_rows(args.data_root/"mfclean"/filename))
    groups,merges=assign_components(records,historical)
    write_json(args.audit_dir/"private_historical_group_merges.json",merges)
    original=build(records,[r["orig"] for r in historical],groups,historical,args.out,args.data_root/"medforge90k_full",args.workers,reuse=args.reuse_mfclean)
    boxes=extension_boxes(args.data_root,args.workspace)
    for path,box in boxes.items():
        if path in records:
            records[path]["box"]=box
    extension=build(records,[p for p,r in records.items() if r["corpus"]=="extension"],groups,[],args.extension_out,args.data_root/"medforge90k_full",args.workers,True)
    write_json(args.audit_dir/"rebuild_summary.json",dict(mfclean=original,extension=extension,merged_historical_components=len(merges)))
    print(json.dumps(dict(mfclean_counts=original["split_audit"]["counts"],extension_counts=extension["split_audit"]["counts"],merged_historical_components=len(merges)),indent=2),flush=True)


if __name__=="__main__":
    main()
