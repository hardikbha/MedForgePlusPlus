#!/usr/bin/env python3
"""Private raw-file identity census. All row-level outputs must remain private.

CPU-only. Rehashes and decodes actual inputs; does not trust prior census hashes.
Never changes raw datasets, checkpoints or historical manifests.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from v3.build_mfclean import UF, known_patient_id


def read_rows(path):
    with open(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_rows(path, records):
    with open(path, "w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def raw_record(job):
    record = dict(job)
    try:
        path = Path(record["path"])
        data = path.read_bytes()
        record.update(content_md5=hashlib.md5(data).hexdigest(), sha256=hashlib.sha256(data).hexdigest(),
                      bytes=len(data), resolved_path=str(path.resolve()))
        with Image.open(io.BytesIO(data)) as image:
            record.update(format=image.format, mode=image.mode, dimensions=list(image.size))
            rgb = np.asarray(image.convert("RGB"))
            record["pixel_sha256"] = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
            gray = image.convert("L")
            thumb = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.uint8)
            tiny = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
            bits = (tiny[:, 1:] > tiny[:, :-1]).flatten()
            record["dhash"] = f"{sum(int(b) << (63-i) for i,b in enumerate(bits)):016x}"
        mask = record.get("mask")
        if mask:
            mask_data = Path(mask).read_bytes()
            record["mask_sha256"] = hashlib.sha256(mask_data).hexdigest()
            with Image.open(io.BytesIO(mask_data)) as image:
                a = np.asarray(image.convert("L")) > 127
            ys, xs = np.where(a)
            record["mask_dimensions"] = [int(a.shape[1]), int(a.shape[0])]
            record["mask_box"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
        return record, thumb
    except Exception as error:
        record["error"] = type(error).__name__ + ": " + str(error)
        return record, np.zeros((32, 32), dtype=np.uint8)


def near_verify(job):
    first, second = job
    try:
        with Image.open(first) as image:
            a = np.asarray(image.convert("RGB"), dtype=np.int16)
        with Image.open(second) as image:
            b = np.asarray(image.convert("RGB"), dtype=np.int16)
        if a.shape != b.shape:
            return first, second, False, {"reason": "different_dimensions"}
        delta = np.abs(a-b)
        mae = float(delta.mean())
        # This is a conservative split-safety component, not an inferred patient ID.
        fraction = float((delta > 10).mean())
        return first, second, mae <= 3 and fraction <= 0.05, {"rgb_mae": mae, "fraction_delta_gt10": fraction}
    except Exception as error:
        return first, second, False, {"reason": str(error)}


def enumerate_inputs(root, workspace):
    mf = root / "medforge90k_full"
    items = json.loads((mf / "dataset_mapping.json").read_text())["items"]
    records = {}
    missing = Counter()
    for item in items:
        path = str(mf / item["image_path"])
        source = item["image_path"] if item["type"] == "real" else item["real_image_path"]
        if not Path(path).is_file():
            missing[item["type"]] += 1
            continue
        records[path] = dict(path=path, corpus="original", organ=item["dataset"], type=item["type"],
                             gen="real" if item["type"] == "real" else item["model"],
                             source=str(mf / source), patient_id=known_patient_id(source, item["dataset"]),
                             mask=str(mf / item["mask_path"]) if item.get("mask_path") else None,
                             disease=item.get("disease"))
    ext = root / "naitik_annotated"
    for subset in ("kidney_normal", "kidney_real_tumor", "kidney_removal", "polyp_normal", "polyp_real", "polyp_removal"):
        for path in sorted((ext / subset / "images").glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            kidney = subset.startswith("kidney")
            source = path if "removal" not in subset else ext / ("kidney_real_tumor" if kidney else "polyp_real") / "images" / path.name
            # Some unresolved polyp filenames carry an explicit unresolved prefix.
            if not source.is_file() and "removal" in subset:
                alternate = source.with_name(source.name.removeprefix("unresolved_"))
                if alternate.is_file():
                    source = alternate
            match = re.match(r"(case_\d{5})_slice_", path.stem)
            records[str(path)] = dict(path=str(path), corpus="extension", subset=subset,
                                     organ="kidney-ct" if kidney else "colon", type="remove" if "removal" in subset else "real",
                                     gen="lama" if "removal" in subset else "real", source=str(source) if source.is_file() else None,
                                     source_link="same_stem_pre_post" if source.is_file() else "unresolved",
                                     patient_id=None, volume_id="kits:" + match.group(1) if match else None)
    for organ, directory, source_dir, suffix in (
        ("colon", root / "colon_injections/images", ext / "polyp_real/images", "_polypinj"),
        ("kidney-ct", workspace / "AAAI/kidney/flux_injected/images", workspace / "AAAI/kidney/normal", "_tumorinj"),
    ):
        sources = {}
        for path in source_dir.iterdir():
            if path.is_file():
                # inject_flux_kidney.py writes source_stem.replace(" ", "_").
                key = path.stem.replace(" ", "_") if organ == "kidney-ct" else path.stem
                if key in sources:
                    raise ValueError("Ambiguous source filename normalization; do not guess a source")
                sources[key] = path
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            source = sources.get(path.stem.removesuffix(suffix))
            records[str(path)] = dict(path=str(path), corpus="extension", subset="colon_inject" if organ == "colon" else "kidney_inject",
                                     organ=organ, type="edit", gen="flux.1-kontext", source=str(source) if source else None,
                                     source_link="generator_preserves_source_stem" if source else "unresolved", patient_id=None)
            if source and str(source) not in records:
                records[str(source)] = dict(path=str(source), corpus="extension_source_only", organ=organ, type="real",
                                             gen="real", source=str(source), patient_id=None)
    # Also decode the actual historical MF-Clean image bytes, not just raw metadata.
    for name in ("train_all", "select_core", "test_core"):
        for row in read_rows(root / "mfclean" / (name + ".jsonl")):
            original = records.get(row["orig"], {})
            records[row["image"]] = dict(path=row["image"], corpus="historical_mfclean", organ=row["organ"],
                                         type=row["type"], gen=row["gen"], source=row["orig"],
                                         patient_id=original.get("patient_id"))
    return records, items, missing


def build_graph(records):
    graph = UF()
    reasons = Counter()
    index = {field: {} for field in ("content_md5", "pixel_sha256", "resolved_path", "patient_id", "volume_id", "study_id", "video_id")}
    for path, record in records.items():
        graph.f(path)
        if record.get("source"):
            graph.u(path, record["source"])
            reasons["source_links"] += 1
        for field, lookup in index.items():
            value = record.get(field)
            if value:
                first = lookup.setdefault(value, path)
                if first != path:
                    graph.u(first, path)
                    reasons[field + "_links"] += 1
    return graph, reasons


def audit_manifest(rows, graph, records):
    identifiers, components, fields = defaultdict(list), defaultdict(set), Counter()
    counts = Counter()
    for row in rows:
        path = row["image"]
        orig = row.get("orig", path)
        component = graph.f(orig)
        identifiers[os.path.realpath(path)].append(row)
        components[component].add(row["split"])
        counts[(row["split"], row.get("organ", "unknown"), row["type"], row.get("gen", "unknown"))] += 1
        record = records.get(orig, {})
        for field in ("patient_id", "source", "content_md5", "pixel_sha256"):
            fields[field] += bool(record.get(field))
    by_image = {row["image"]: row for row in rows}
    twins = Counter()
    for row in rows:
        if row["type"] == "real":
            continue
        twin = row.get("twin")
        target = by_image.get(twin)
        source = records.get(row.get("orig", ""), {}).get("source")
        if not twin:
            twins["missing"] += 1
        elif not target:
            twins["dangling"] += 1
        elif target["type"] != "real" or target["split"] != row["split"]:
            twins["invalid_partition_or_type"] += 1
        elif graph.f(target.get("orig", target["image"])) != graph.f(row.get("orig", row["image"])):
            twins["cross_component"] += 1
        elif target.get("orig") != source:
            twins["substituted_representative"] += 1
        else:
            twins["exact_source"] += 1
    return {"rows": len(rows), "repeated_resolved_image_paths": sum(len(x)-1 for x in identifiers.values()),
            "cross_split_components": sum(len(x)>1 for x in components.values()),
            "identity_coverage_rows": dict(fields), "twins": dict(twins),
            "counts": [dict(split=a, organ=b, type=c, generator=d, rows=n) for (a,b,c,d),n in sorted(counts.items())]}, components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--historical-hashes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--census-only", action="store_true", help="Save complete rehashed/decoded checkpoint; finish with resume_identity_audit.py")
    args = parser.parse_args()
    if not 1 <= args.workers <= 7:
        parser.error("Use 1–7 workers within the existing eight-CPU allocation")
    args.out.mkdir(parents=True, exist_ok=False)
    os.chmod(args.out, 0o700)
    inputs, items, missing = enumerate_inputs(args.data_root, args.workspace)
    old = {}
    with args.historical_hashes.open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            old[row["path"]] = row["md5"]
    current, thumbs = {}, []
    mismatch = Counter()
    with ProcessPoolExecutor(args.workers) as pool, (args.out / "raw_records.jsonl").open("w") as output:
        for n, (record, thumb) in enumerate(pool.map(raw_record, inputs.values(), chunksize=16), 1):
            path = record["path"]
            if path in old:
                mismatch["compared"] += 1
                mismatch["changed"] += old[path] != record.get("content_md5")
            else:
                mismatch["not_in_historical_census"] += 1
            record["thumb_index"] = len(thumbs)
            current[path] = record
            thumbs.append(thumb)
            output.write(json.dumps(record) + "\n")
            if n % 10000 == 0:
                print(f"raw census {n}/{len(inputs)}", flush=True)
    thumbnails = np.stack(thumbs)
    np.save(args.out / "thumbs32.npy", thumbnails)
    if args.census_only:
        print(json.dumps({"census_checkpoint_images":len(current),"raw_errors":sum(bool(r.get('error')) for r in current.values()),"next":"resume_identity_audit.py on this output directory"}),flush=True)
        return
    graph, reasons = build_graph(current)
    # Five exact bit blocks guarantee a shared block for any 64-bit dHash distance <=4.
    blocks = defaultdict(list)
    candidates = set()
    seen_pairs = 0
    paths = list(current)
    for i, path in enumerate(paths):
        row = current[path]
        if row.get("error"):
            continue
        value = int(row["dhash"], 16)
        hits = set()
        for shift, width in ((0,13), (13,13), (26,13), (39,13), (52,12)):
            key = (row["organ"], shift, (value >> shift) & ((1<<width)-1))
            hits.update(blocks[key])
            blocks[key].append(i)
        for j in hits:
            other = current[paths[j]]
            if graph.f(path) == graph.f(paths[j]):
                continue
            if (value ^ int(other["dhash"],16)).bit_count() > 4:
                continue
            seen_pairs += 1
            if np.abs(thumbnails[i].astype(np.int16)-thumbnails[j].astype(np.int16)).mean() <= 3:
                candidates.add((paths[j],path))
    print(f"near candidates across source/patient/content components: {len(candidates)}",flush=True)
    accepted, rejected = 0, 0
    with ProcessPoolExecutor(args.workers) as pool, (args.out / "near_duplicate_decisions.jsonl").open("w") as output:
        for first, second, match, metrics in pool.map(near_verify, sorted(candidates), chunksize=16):
            output.write(json.dumps(dict(first=first,second=second,accepted=match,**metrics))+"\n")
            if match:
                graph.u(first,second)
                accepted += 1
            else:
                rejected += 1
    component_rows=[]
    for path,row in current.items():
        row["component"] = "component:" + hashlib.sha256(graph.f(path).encode()).hexdigest()[:24]
        component_rows.append(row)
    write_rows(args.out / "identity_records.jsonl",component_rows)
    manifest_rows=[]
    for name in ("train_all", "select_core", "test_core"):
        manifest_rows.extend(read_rows(args.data_root / "mfclean" / (name+".jsonl")))
    original_report, components = audit_manifest(manifest_rows,graph,current)
    extension_rows=[]
    for row in read_rows(args.data_root / "circe_eval_split_v2.jsonl"):
        source=current.get(row["image"],{})
        extension_rows.append(dict(row,split="test",organ=source.get("organ","unknown"),gen=source.get("gen","unknown")))
    legacy_eval={r["image"] for r in extension_rows}
    legacy_overlap={}
    for filename in ("circe_sft_v2.jsonl","circe_sft_v3.jsonl"):
        paths_sft={r["images"][0] for r in read_rows(args.data_root/filename)}
        train_components={graph.f(q) for q in paths_sft}
        legacy_overlap[filename]={"unique_images":len(paths_sft),"exact_eval_paths":len(legacy_eval&paths_sft),
                                  "eval_components_exposed":sum(graph.f(p) in train_components for p in legacy_eval)}
    report={"schema":"medforge-private-identity-audit-v2","mapping_items":len(items),"missing_mapping_images_by_type":dict(missing),
            "raw_files_rehashed_decoded":len(current),"raw_errors":sum(bool(r.get('error')) for r in current.values()),
            "historical_hash_comparison":dict(mismatch),"graph_link_counts":dict(reasons),
            "near_duplicates":{"dhash_cross_component_candidates":seen_pairs,"thumbnail_candidates":len(candidates),"full_rgb_verified_links":accepted,"rejected":rejected,
                               "rule":"same organ and dimensions; dHash Hamming<=4; thumbnail MAE<=3; full RGB MAE<=3 and <=5% channel values differ by >10"},
            "historical_mfclean":original_report,"extension_historical_eval_rows":len(extension_rows),"extension_legacy_overlap":legacy_overlap,
            "identity_limitations":{"patient":"Only known filename conventions; unknown remains null","study":"unavailable","video":"Kvasir/WCE video/patient linkage unavailable","kidney_case":"KiTS case parsed; Kaggle kidney patient IDs unavailable","near_duplicate_meaning":"conservative split-safety components, not inferred patient identities"}}
    write_json(args.out/"aggregate_audit.json",report)
    print(json.dumps({"raw_files":len(current),"raw_errors":report["raw_errors"],"historical_mfclean_cross_split_components":original_report["cross_split_components"],"extension_eval":len(extension_rows)}),flush=True)


if __name__=="__main__":
    main()
