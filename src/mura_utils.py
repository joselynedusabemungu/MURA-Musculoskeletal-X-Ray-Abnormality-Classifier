from pathlib import Path
import json, random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

SEED=42
IMAGE_SIZE=(128,128)
CLASS_NAMES=['Normal','Abnormal']

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)

def find_mura_root(project_root=None):
    p=Path(project_root or Path.cwd()).resolve()
    candidates=[p/'MURA-v1.1',p/'MURA',p/'data'/'MURA-v1.1',p/'data'/'MURA']
    candidates += list(p.glob('**/MURA-v1.1')) + list(p.glob('**/MURA'))
    for x in candidates:
        if (x/'train').is_dir() and (x/'valid').is_dir(): return x
    raise FileNotFoundError('Place MURA-v1.1 beside the notebook/project or under data/MURA-v1.1.')

def build_index(root, split):
    rows=[]; base=Path(root)/split
    for p in sorted(base.glob('XR_*/*/study*_*/**/*.png')):
        parts=p.relative_to(base).parts; study=parts[2]
        label=int(study.lower().endswith('positive'))
        rows.append({'path':str(p.resolve()),'split':split,'body_part':parts[0].replace('XR_','').replace('_',' ').title(),'patient_id':parts[1],'study_id':study,'label':label,'class_name':CLASS_NAMES[label]})
    if not rows: raise FileNotFoundError(f'No MURA PNGs found below {base}.')
    return pd.DataFrame(rows)

def make_splits(train_df, valid_df, val_fraction=.15, seed=SEED):
    rng=np.random.default_rng(seed); patients=train_df.patient_id.drop_duplicates().to_numpy(); rng.shuffle(patients)
    n=max(1,int(len(patients)*val_fraction)); vp=set(patients[:n])
    tr=train_df[~train_df.patient_id.isin(vp)].copy(); va=train_df[train_df.patient_id.isin(vp)].copy(); te=valid_df.copy()
    assert set(tr.patient_id).isdisjoint(va.patient_id) and set(tr.patient_id).isdisjoint(te.patient_id) and set(va.patient_id).isdisjoint(te.patient_id)
    return tr.reset_index(drop=True),va.reset_index(drop=True),te.reset_index(drop=True)



def extract_features(path_or_image):
    if isinstance(path_or_image,(str,Path)):
        with Image.open(path_or_image) as im: im=im.convert('L')
    else: im=path_or_image.convert('L')
    im=im.resize(IMAGE_SIZE)
    arr=np.asarray(im,dtype=np.float32)/255.0
    return arr.reshape(-1)

extract_hog = extract_features

def make_features(frame, limit=None, progress_every=500):
    f=frame if limit is None else frame.head(limit)
    X=[]
    for i,p in enumerate(f.path):
        X.append(extract_features(p))
        if progress_every and (i+1)%progress_every==0: print(f'Extracted {i+1}/{len(f)}')
    return np.asarray(X), f.label.to_numpy(), f.path.to_numpy()

def build_model():
    return Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=1000,class_weight='balanced',solver='liblinear',random_state=SEED))])

def predict_image(model,image):
    x=extract_features(image).reshape(1,-1); prob=model.predict_proba(x)[0]; idx=int(np.argmax(prob))
    return {'label':CLASS_NAMES[idx],'confidence':float(prob[idx]),'probabilities':dict(zip(CLASS_NAMES,prob.astype(float)))}

def save_bundle(model,path,metadata): joblib.dump({'model':model,'metadata':metadata},path)
def load_bundle(path): return joblib.load(path)
