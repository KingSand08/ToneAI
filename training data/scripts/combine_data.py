import os, sys, argparse, pathlib
from collections import OrderedDict

try:
    import pandas as pd  # requires openpyxl for xlsx I/O
except Exception as e:
    sys.stderr.write('[!] pandas/openpyxl required. Try: pip install pandas openpyxl\n')
    raise

# Known output schema (order matters)
FINAL_COLS = [
    'Id', 'File', 'Dataset',
    'Neutral','Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation',
    'I1', 'I2', 'I3',
]

EMO_INT_COLS = ['Neutral','Joy','Trust','Fear','Surprise','Sadness','Disgust','Anger','Anticipation','I1','I2','I3']

def read_xlsx(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        sys.stderr.write(f'[!] Missing input: {path}\n')
        sys.exit(2)
    return pd.read_excel(path)

def ensure_columns(df: pd.DataFrame, default_Dataset: str) -> pd.DataFrame:
    # Add missing columns with sensible defaults
    if 'Dataset' not in df.columns:
        df['Dataset'] = default_Dataset
    if 'File' not in df.columns:
        df['File'] = ''
    for c in EMO_INT_COLS:
        if c not in df.columns:
            df[c] = 0
    if 'Id' not in df.columns:
        df['Id'] = range(len(df))
    # Keep only the known columns (extras are dropped), and in the desired order
    for c in FINAL_COLS:
        if c not in df.columns:
            # create any remaining missing with neutral defaults
            df[c] = 0 if c in EMO_INT_COLS or c in ('I1','I2','I3','Id') else ''
    return df[FINAL_COLS]

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for c in EMO_INT_COLS + ['I1','I2','I3']:
        df[c] = df[c].fillna(0).astype(int)
    df['Id'] = df['Id'].fillna(0).astype(int)
    df['File'] = df['File'].astype(str)
    df['Dataset'] = df['Dataset'].astype(str)
    return df

def main():
    script_dir = pathlib.Path(__file__).resolve().parent
    base_dir   = script_dir.parent
    cat_dir    = base_dir / 'categories'

    parser = argparse.ArgumentParser()
    parser.add_argument('--crema', default=str(cat_dir / 'crema-test.xlsx'))
    parser.add_argument('--emo',   default=str(cat_dir / 'emo-test.xlsx'))
    parser.add_argument('--out',   default=str(cat_dir / 'data.xlsx'))
    args = parser.parse_args()

    crema_path = pathlib.Path(args.crema)
    emo_path   = pathlib.Path(args.emo)
    out_path   = pathlib.Path(args.out)

    crema = read_xlsx(crema_path)
    emo   = read_xlsx(emo_path)

    # Ensure schema & Dataset labels (fallbacks if missing)
    crema = ensure_columns(crema, default_Dataset='CREMA-D')
    emo   = ensure_columns(emo,   default_Dataset='EmoGator')

    # Coerce types
    crema = coerce_types(crema)
    emo   = coerce_types(emo)

    # Combine
    combined = pd.concat([crema, emo], ignore_index=True)

    # Drop exact duplicates by Dataset+File to be safe
    if {'Dataset','File'}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=['Dataset','File'], keep='first').reset_index(drop=True)

    # Reassign Id to be contiguous 0..N-1
    combined['Id'] = range(len(combined))

    # Final column order & types
    combined = combined[FINAL_COLS]
    combined = coerce_types(combined)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_excel(out_path, index=False, engine='openpyxl')
    print(f'[OK] Wrote {out_path} with {len(combined)} rows.')

if __name__ == '__main__':
    main()
