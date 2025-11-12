#!/usr/bin/env python3
import os, re, sys, argparse, pathlib
from collections import OrderedDict

# --- args & defaults (CLI > env > defaults from script location) ---
script_dir = pathlib.Path(__file__).resolve().parent
base_dir   = script_dir.parent
def_data   = base_dir / "files" / "EmoGator-main" / "data" / "mp3"
def_out    = base_dir / "categories" / "emo-test.xlsx"

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=os.environ.get("DATA_DIR_PY", str(def_data)))
p.add_argument("--out-xlsx", default=os.environ.get("OUT_XLSX_PY", str(def_out)))
a = p.parse_args()

DATA_DIR = pathlib.Path(a.data_dir)
OUT_XLSX = pathlib.Path(a.out_xlsx)

# --- deps ---
try:
    import pandas as pd  # requires openpyxl for xlsx
except Exception as e:
    sys.stderr.write("[!] pandas/openpyxl required. Try: pip install pandas openpyxl\n")
    raise

if not DATA_DIR.exists():
    sys.stderr.write(f"[!] Data directory not found: {DATA_DIR}\n")
    sys.exit(2)

# EmoGator filename: NNNNNN-EE-I.mp3 (id-emotion-intensity)
# EE codes (01-30) map to different emotions
# Mapping EmoGator emotions to standard 8 emotion categories (consistent with CREMA-D)
# Based on EmoGator dataset documentation:
# 01-05: Anger, 06-10: Disgust, 11-15: Fear, 16-20: Joy, 21-25: Neutral, 26-30: Sadness

['Adoration', 'Amusement', 'Anger', 'Awe', 'Confusion', 'Contempt', 'Contentment', 'Desire', 'Disappointment', 'Disgust', 'Distress', 'Ecstasy', 'Elation', 'Embarrassment', 'Fear', 'Guilt', 'Interest', 'Neutral', 'Pain', 'Pride', 'Realization', 'Relief', 'Romantic Love', 'Sadness', 'Serenity', 'Shame', 'Surprise (Negative)', 'Surprise (Positive)', 'Sympathy', 'Triumph']
#  01: {Ecstasy = [Joy, i3]} = Adoration
#  02: {Joy = [Joy, i2]} = Amusement
#  03: {Anger = [Anger, i2]} = Anger
#! 04: Submissive = Awe (surprise/fear)
#! 05: Surprise/Anticipation = Confusion
#! 06: Contempt = Contempt (anger/disgust)
#  07: {Acceptance = [Trust, i1]} = Contentment
#! 08: Anticipation/Interest = Desire
#! 09: Disapproval  = Disappointment (sadness/surprise)
#  10: {Disgust = [Disgust, i2]} = Disgust
#  11: {Apprehension = [Fear, i1]} = Distress
#  12: {Ecstasy = [Joy, i3]} = Ecstasy
#! 13: Joy/Ecstasy (4th layer blend) = Elation
#! 14: Fear/Disgust = Embarrassment
#  15: {Fear = [Fear, i2]} = Fear
#  16: {Grief = [Sad, i3]} = Guilt
#  17: {Interest = [Anticipation, i1]} = Interest
#  18: {Neutral = [Neutral, i2]} = Neutral
#  19: {Terror = [Fear, i3]} = Pain
#! 20: Aggressiveness = Pride (annoyance/boredom)
#  21: {Trust = [Trust, i2]} = Realization
#! 22: Joy/Trust = Relief
#! 23: Love = Romantic Love (ecstasy/admiration), Relief (trust/joy)
#  24: {Sad = [Sad, i2]} = Sadness
#  25: {Serenity = [Joy, i1]} = Serenity
#! 26: Remorse = Shame (loathing/grief)
#  27: {Surprise = [Surprise, i2]} = Surprise (Negative)
#  28: {Surprise = [Surprise, i2]} = Surprise (Positive)
#! 29: Optimism = Sympathy (interest/serenity)
#  30: {Admiration = [Trust, i3]} = Triumph
#! Anticipation = x
#! Amazement = x
#! Loathing = x
#! Rage = x
#! Vigilance = x
#! Distraction = x
#! Pensiveness = x
#! Boredom = x
#! Annoyance = x
emogator_to_our = {
    # Joy
    '01': 'Joy', '02': 'Joy', '12': 'Joy', '25': 'Joy',
    # Trust
    '07': 'Trust', '21': 'Trust', '30': 'Trust',
    # Fear
    '11': 'Fear', '15': 'Fear', '19': 'Fear',
    # Surprise
    '27': 'Surprise', '28': 'Surprise',
    # Sad
    '16': 'Sad', '24': 'Sad',
    # Disgust
    '10': 'Disgust',
    # Anger
    '03': 'Anger',
    # Anticipation
    '17': 'Anticipation',
}

# Intensity mapping: last digit (1, 2, 3)
# 1 = Low intensity (I1), 2 = Medium intensity (I2), 3 = High intensity (I3)
inten_to_ix = {'1': 'I1', '2': 'I2', '3': 'I3'}

# EmoGator filename pattern: NNNNNN-EE-I.mp3
pat = re.compile(r'^(?P<id>\d{6})-(?P<emo>\d{2})-(?P<inten>\d)\.mp3$', re.I)

columns = [
    'Id','dataset','File',
    'Joy','Trust','Fear','Surprise','Sadness','Disgust','Anger','Anticipation',
    'I1','I2','I3',
]

# collect files
files = []
for root, _, fnames in os.walk(DATA_DIR):
    for fn in fnames:
        lf = fn.lower()
        if lf.endswith(('.wav','.mp3','.flv')):
            files.append(os.path.join(root, fn))
files.sort()

rows = []
rid = 0
for abspath in files:
    fname = os.path.basename(abspath)
    m = pat.match(fname)
    if not m:
        continue
    emo = m.group('emo')
    inten = m.group('inten')

    row = OrderedDict((c, 0) for c in columns)
    row['dataset'] = 'EmoGator'
    row['Id'] = rid
    row['File'] = str(pathlib.Path(abspath).relative_to(DATA_DIR))

    our = emogator_to_our.get(emo)
    if our in ('Joy','Trust','Fear','Surprise','Sadness','Disgust','Anger','Anticipation'):
        row[our] = 1

    ix = inten_to_ix.get(inten)
    if ix:
        row[ix] = 1

    rows.append(row)
    rid += 1

df = pd.DataFrame(rows, columns=columns)
OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
df.to_excel(OUT_XLSX, index=False, engine='openpyxl')
print(f"[OK] Wrote {OUT_XLSX} with {len(df)} rows.")
