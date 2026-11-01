"""Part 4: explanation-text / template audit. Read-only."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, re, random, collections, sys
ROOT = _release_path('AAAI/MedForge++/medforge90k_full')
PP = _release_path('AAAI/MedForge++')
rng = random.Random(7)
out = {}

# ---------- MF90K explanation files ----------
d = json.load(open(os.path.join(ROOT, 'dataset_mapping.json')))
items = d['items']
fk = [x for x in items if x['type'] != 'real']
rl = [x for x in items if x['type'] == 'real']
samp_f = rng.sample(fk, 1500)
samp_r = rng.sample(rl, 300)
miss_f = sum(not os.path.exists(os.path.join(ROOT, x['explain_path'])) for x in samp_f if x['explain_path'])
miss_r = sum(not (x['explain_path'] and os.path.exists(os.path.join(ROOT, x['explain_path']))) for x in samp_r)
out['mf90k_explain_missing_fake_frac_of_1500'] = miss_f / 1500
out['mf90k_explain_missing_real_frac_of_300'] = miss_r / 300
miss_by_model = collections.Counter(); tot_by_model = collections.Counter()
for x in samp_f:
    tot_by_model[x['model']] += 1
    if not os.path.exists(os.path.join(ROOT, x['explain_path'])):
        miss_by_model[x['model']] += 1
out['explain_missing_by_model'] = {m: f'{miss_by_model[m]}/{tot_by_model[m]}' for m in sorted(tot_by_model)}

texts = []
for x in samp_f:
    p = os.path.join(ROOT, x['explain_path'])
    if os.path.exists(p):
        texts.append((x, open(p, errors='replace').read()))
print('read', len(texts), file=sys.stderr)

# sentence-level duplication
sent_counter = collections.Counter()
per_doc_sents = []
for x, t in texts:
    body = re.sub(r'x1="\d+" y1="\d+" x2="\d+" y2="\d+"', 'BOX', t)
    ss = [s.strip(' -*\n') for s in re.split(r'(?<=[.!?])\s+|\n', body) if len(s.strip()) > 20]
    per_doc_sents.append(ss)
    sent_counter.update(set(ss))
n = len(texts)
tot_s = sum(len(s) for s in per_doc_sents)
dup_s = sum(sum(1 for s in ss if sent_counter[s] > 1) for ss in per_doc_sents)
out['n_explain_texts'] = n
out['mean_words'] = sum(len(t.split()) for _, t in texts) / n
out['frac_sentences_shared_with_another_doc'] = dup_s / tot_s
out['top_shared_sentences'] = [(s[:140], c) for s, c in sent_counter.most_common(12)]
# phrase frequencies
phr = ['unnaturally smooth', 'smooth', 'noise', 'sticker', 'blurr', 'inpaint', 'texture', 'sharp', 'boundar', 'halo',
       'lack', 'confirms that this image is a deepfake', 'anatomical errors present in this image are as follows']
out['phrase_doc_freq'] = {p: sum(p.lower() in t.lower() for _, t in texts) / n for p in phr}
# does the explanation say "removed/erased/absent" for removal and "added/inserted/lesion" for edit?
def has(t, ws):
    t = t.lower(); return any(w in t for w in ws)
rem_words = ['remov', 'erase', 'absent', 'missing', 'painted over', 'obliterat']
add_words = ['insert', 'implant', 'added', 'pasted', 'overlaid', 'superimpos']
for ty in ['edit', 'remove']:
    T = [t for x, t in texts if x['type'] == ty]
    out[f'{ty}_n'] = len(T)
    out[f'{ty}_mentions_removal_words'] = sum(has(t, rem_words) for t in T) / max(1, len(T))
    out[f'{ty}_mentions_insertion_words'] = sum(has(t, add_words) for t in T) / max(1, len(T))
# box in text vs mask bbox
from PIL import Image
import numpy as np
agree = []; ious = []
for x, t in texts[:400]:
    m = re.search(r'x1="(\d+)" y1="(\d+)" x2="(\d+)" y2="(\d+)"', t)
    mp = os.path.join(ROOT, x['mask_path'])
    if not m or not os.path.exists(mp):
        continue
    bx = list(map(int, m.groups()))
    a = np.asarray(Image.open(mp).convert('L')) > 127
    if not a.any():
        continue
    ys, xs = np.where(a)
    mb = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    ix = max(0, min(bx[2], mb[2]) - max(bx[0], mb[0])); iy = max(0, min(bx[3], mb[3]) - max(bx[1], mb[1]))
    inter = ix * iy
    u = (bx[2] - bx[0]) * (bx[3] - bx[1]) + (mb[2] - mb[0]) * (mb[3] - mb[1]) - inter
    ious.append(inter / u if u else 0)
out['explain_box_vs_mask_iou_n'] = len(ious)
out['explain_box_vs_mask_iou_mean'] = float(np.mean(ious)) if ious else None
out['explain_box_vs_mask_iou_ge0.9'] = float(np.mean(np.array(ious) >= 0.9)) if ious else None
# examples (2 per type) for the report
out['examples'] = []
for ty in ['edit', 'remove']:
    for x, t in [z for z in texts if z[0]['type'] == ty][:2]:
        out['examples'].append({'key': x['key'], 'text': t[:1200]})

# ---------- our SFT template annotations ----------
def strip_vars(s):
    s = re.sub(r'x1="\d+" y1="\d+" x2="\d+" y2="\d+"', 'BOX', s)
    s = re.sub(r'<description>.*?</description>', '<description>D</description>', s, flags=re.S)
    return s

for fn in ['mfp_train_crossforgery.jsonl', 'mfp_v2_train_crossforgery_strict.jsonl', 'mfp_v2_replay_crossforgery_strict.jsonl',
           'mfp_train_indomain.jsonl']:
    p = os.path.join(PP, fn)
    if not os.path.exists(p):
        continue
    templ = collections.Counter(); cue = collections.Counter(); empty_ev = 0; n = 0
    by_label = collections.defaultdict(collections.Counter)
    desc = collections.Counter(); nturns = collections.Counter(); repair = collections.Counter()
    for line in open(p):
        r = json.loads(line)
        msgs = r['messages']
        asst = [m['content'] for m in msgs if m['role'] == 'assistant']
        nturns[len(asst)] += 1
        a0 = asst[0]
        n += 1
        img = r['images'][0]
        lab = 'real' if '/real/' in img else ('remove' if '-remove' in img or 'removal' in img else ('edit' if '-edit' in img or 'inj' in img else 'other'))
        t = strip_vars(a0)
        templ[t] += 1
        by_label[lab][t] += 1
        for c in re.findall(r'<evidence cue="([^"]*)">(.*?)</evidence>', a0, flags=re.S):
            cue[c[0]] += 1
            if not c[1].strip():
                empty_ev += 1
        m = re.search(r'<description>(.*?)</description>', a0, flags=re.S)
        if m:
            desc[m.group(1)] += 1
        if len(asst) > 1:
            repair[strip_vars(asst[1])] += 1
    out[fn] = {'n': n, 'n_unique_assistant_texts_after_box_desc_masking': len(templ),
               'top_template_share': templ.most_common(1)[0][1] / n,
               'unique_templates_by_label': {k: len(v) for k, v in by_label.items()},
               'n_by_label': {k: sum(v.values()) for k, v in by_label.items()},
               'evidence_cues': dict(cue), 'empty_evidence_tags': empty_ev,
               'n_unique_descriptions': len(desc), 'top_descriptions': desc.most_common(6),
               'assistant_turns': dict(nturns),
               'n_unique_repair_turns': len(repair),
               'top_repair': [(k[:300], v) for k, v in repair.most_common(3)]}
    # dump one template per label
    out[fn]['templates'] = {k: v.most_common(2) for k, v in by_label.items()}

json.dump(out, open(_release_path('CIRCE/v3_audit/labels_text_audit.json'), 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != 'examples'}, indent=1, default=str)[:12000])
