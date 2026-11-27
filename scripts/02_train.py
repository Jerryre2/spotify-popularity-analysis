# Standalone Python version of the corresponding analysis notebook.
# Run via: python run.py train --data /path/to/archive.zip
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = Path(os.environ.get("SPOTIFY_OUTPUT_DIR", str(PROJECT_ROOT / "runs")))
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

def display(value):
    """Render notebook tables as text; figures are saved by the analysis itself."""
    if hasattr(value, "to_string"):
        print(value.to_string())
    else:
        print(value)

def main():
    from pathlib import Path
    from zipfile import ZipFile
    from functools import lru_cache
    from itertools import combinations
    import ast
    import hashlib
    import json
    import platform
    import time
    import warnings
    import numpy as np
    import pandas as pd
    import sklearn
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
        precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score)
    from sklearn.model_selection import train_test_split
    from sklearn.exceptions import ConvergenceWarning

    SEED = 20261123
    THRESHOLD = 60
    N_JOBS = int(os.environ.get("SPOTIFY_JOBS", "2"))
    DATA_PATH = Path(os.environ.get("SPOTIFY_DATA_PATH", str(PROJECT_ROOT / "data" / "archive.zip")))
    OUT = RESULT_ROOT / 'model_results'
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO = ['danceability','energy','tempo','loudness','valence','acousticness','instrumentalness','speechiness','liveness']
    FEATURES = AUDIO + ['duration_ms','key','mode','time_signature']
    CONTINUOUS = [c for c in AUDIO if c!='tempo'] + ['duration_ms']
    CATEGORICAL = ['key','mode']
    versions = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'scikit-learn': sklearn.__version__, 'matplotlib': matplotlib.__version__, 'joblib': joblib.__version__}
    print('Environment:', versions)
    with ZipFile(DATA_PATH) as z:
        digest = hashlib.sha256()
        with z.open('tracks.csv') as f:
            for block in iter(lambda:f.read(1024*1024), b''): digest.update(block)
        with z.open('tracks.csv') as f: raw = pd.read_csv(f, keep_default_na=False)
    assert digest.hexdigest() == '8a08c199bd93438783e915470ec3ea1142c1267f2da48b7d1315d2b9f5d39cf9', 'Dataset differs from the audited version; review before using fixed conclusions.'
    assert raw.id.is_unique
    assert np.isfinite(raw[FEATURES+['popularity']].to_numpy()).all()
    assert raw.popularity.between(0,100).all() and raw.popularity.mod(1).eq(0).all()
    assert not set(['popularity','high_popularity','release_year','artists','id_artists','explicit']) & set(FEATURES)
    y = raw.popularity.ge(THRESHOLD).astype('int8').to_numpy()
    X = raw[FEATURES].copy()
    # Zero handling is contained inside the fitted pipeline, including saved models.
    print('Rows retained:', len(raw), '| positive rows:', int(y.sum()))
    print('Zero tempo -> training median with indicator:', int(X.tempo.eq(0).sum()))
    print('Zero time signature -> unknown category:', int(X.time_signature.eq(0).sum()))

    @lru_cache(maxsize=None)
    def parse_ids(s):
        value = ast.literal_eval(s)
        assert isinstance(value,list) and value and all(isinstance(v,str) and v.strip() for v in value)
        return tuple(sorted(set(value)))
    artist_lists = raw.id_artists.map(parse_ids)
    parent = np.arange(len(raw), dtype=np.int64)
    sizes = np.ones(len(raw), dtype=np.int64)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)
    def union(a,b):
        a,b = find(a),find(b)
        if a == b: return
        if sizes[a]<sizes[b]: a,b=b,a
        parent[b]=a
        sizes[a]+=sizes[b]
    def connect_codes(codes):
        first = {}
        for i,c in enumerate(codes):
            if c<0: continue
            if c in first: union(first[c],i)
            else: first[c]=i
    normalized_names = raw.name.str.strip().str.replace(r'\s+',' ',regex=True).str.casefold()
    name_keys = pd.Series([(n,a) if n else None for n,a in zip(normalized_names,artist_lists)])
    name_codes,_ = pd.factorize(name_keys, sort=False)
    connect_codes(name_codes)
    feature_codes = raw.groupby(FEATURES, sort=False, dropna=False).ngroup().to_numpy()
    connect_codes(feature_codes)
    roots = np.array([find(i) for i in range(len(raw))])
    canonical_ids = pd.Series(raw.id.to_numpy()).groupby(roots).min()
    version_group = pd.Series(roots).map(canonical_ids).to_numpy()
    print('Version groups:', len(canonical_ids), '| largest group:', int(pd.Series(version_group).value_counts().iloc[0]))

    @lru_cache(maxsize=None)
    def hash_partition(key, namespace):
        h = hashlib.blake2b(f'{SEED}|{namespace}|{key}'.encode(), digest_size=8).digest()
        u = int.from_bytes(h, 'big') / 2**64
        return 'train' if u<.6 else ('validation' if u<.8 else 'test')
    version_split = np.array([hash_partition(g,'version') for g in version_group],dtype=object)
    artist_split = []
    for ids in artist_lists:
        parts = {hash_partition(a,'artist') for a in ids}
        artist_split.append(next(iter(parts)) if len(parts)==1 else 'excluded_cross_artist')
    artist_split = np.array(artist_split,dtype=object)
    kept = np.isin(artist_split,['train','validation','test'])
    sp = pd.DataFrame({'group':version_group[kept], 'partition':artist_split[kept]})
    conflict_groups = set(sp.groupby('group').partition.nunique().loc[lambda s:s>1].index)
    conflicting = kept & pd.Series(version_group).isin(conflict_groups).to_numpy()
    artist_split[conflicting] = 'excluded_cross_version'
    splits = {'version_grouped': version_split, 'artist_disjoint': artist_split}

    split_rows=[]
    overlap_rows=[]
    for experiment, parts in splits.items():
        for part in pd.unique(parts):
            idx=np.flatnonzero(parts==part)
            split_rows.append({'experiment':experiment,'partition':part,'rows':len(idx),
                               'positives':int(y[idx].sum()),'positive_rate':float(y[idx].mean()),
                               'multi_artist_rate':float(artist_lists.iloc[idx].map(len).gt(1).mean())})
        idxs={p:np.flatnonzero(parts==p) for p in ['train','validation','test']}
        artist_sets={p:{a for ids in artist_lists.iloc[ix] for a in ids} for p,ix in idxs.items()}
        for a,b in combinations(idxs,2):
            shared_versions = len(set(version_group[idxs[a]]) & set(version_group[idxs[b]]))
            shared_artists = len(artist_sets[a] & artist_sets[b])
            overlap_rows.append({'experiment':experiment,'pair':f'{a}/{b}',
                                 'shared_version_groups':shared_versions,'shared_artists':shared_artists})
            assert shared_versions==0
            if experiment=='artist_disjoint': assert shared_artists==0
        assert sum(len(v) for v in idxs.values()) == int(np.isin(parts,list(idxs)).sum())
    split_summary = pd.DataFrame(split_rows)
    overlap_summary = pd.DataFrame(overlap_rows)
    display(split_summary.round(5))
    display(overlap_summary)
    manifest = pd.DataFrame({'id':raw.id, 'version_group':version_group,
                             'version_grouped_split':version_split,'artist_disjoint_split':artist_split})
    manifest.to_csv(OUT/'split_manifest.csv.gz',index=False,compression='gzip')
    split_summary.to_json(OUT/'split_summary.json',orient='records',indent=2)
    overlap_summary.to_json(OUT/'overlap_checks.json',orient='records',indent=2)

    def make_preprocess():
        return ColumnTransformer([
            ('numeric', Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),
                                  ('scale',StandardScaler())]),CONTINUOUS),
            ('tempo', Pipeline([('impute',SimpleImputer(missing_values=0,strategy='median',add_indicator=True)),
                                ('scale',StandardScaler())]),['tempo']),
            ('categorical',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value=-1)),
                                     ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CATEGORICAL),
            ('time_signature',Pipeline([('impute',SimpleImputer(missing_values=0,strategy='constant',fill_value=-1)),
                                       ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),['time_signature'])
        ],remainder='drop')
    def candidates():
        return {
            'logistic_regression': LogisticRegression(C=1,class_weight='balanced',max_iter=1500,random_state=SEED),
            'forest_depth12': RandomForestClassifier(n_estimators=160,max_depth=12,min_samples_leaf=20,
                max_samples=.65,class_weight='balanced_subsample',max_features='sqrt',n_jobs=N_JOBS,random_state=SEED),
            'forest_depth20': RandomForestClassifier(n_estimators=160,max_depth=20,min_samples_leaf=5,
                max_samples=.65,class_weight='balanced_subsample',max_features='sqrt',n_jobs=N_JOBS,random_state=SEED)
        }
    def choose_threshold(labels,scores):
        p,r,t=precision_recall_curve(labels,scores)
        f=np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],out=np.zeros(len(t)),where=(p[:-1]+r[:-1])>0)
        # Last maximum chooses the higher threshold if tied.
        j=np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]
        return float(t[j]),float(f[j])
    fitted={}
    validation_rows=[]
    decisions={}
    for experiment,parts in splits.items():
        train=np.flatnonzero(parts=='train'); val=np.flatnonzero(parts=='validation')
        fitted[experiment]={}
        print(f'BEGIN {experiment}: train={len(train)}, validation={len(val)}',flush=True)
        for name,estimator in candidates().items():
            started=time.monotonic()
            model=Pipeline([('preprocess',make_preprocess()),('classifier',estimator)])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                model.fit(X.iloc[train],y[train])
            convergence=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
            assert not convergence, convergence
            scores=model.predict_proba(X.iloc[val])[:,1]
            threshold,val_f1=choose_threshold(y[val],scores)
            ap=float(average_precision_score(y[val],scores))
            fitted[experiment][name]={'model':model,'threshold':threshold,'validation_ap':ap}
            row={'experiment':experiment,'candidate':name,'validation_ap':ap,
                 'validation_f1':val_f1,'threshold':threshold,'fit_and_validate_seconds':time.monotonic()-started}
            validation_rows.append(row)
            print(f'DONE {name}: validation AP={ap:.5f}, threshold={threshold:.5f}, seconds={row["fit_and_validate_seconds"]:.1f}',flush=True)
        best_forest=max(['forest_depth12','forest_depth20'],key=lambda n:fitted[experiment][n]['validation_ap'])
        winner=max(['logistic_regression',best_forest],key=lambda n:fitted[experiment][n]['validation_ap'])
        decisions[experiment]={'selected_forest':best_forest,'selected_model':winner,
                               'selected_threshold':fitted[experiment][winner]['threshold']}
    validation_results=pd.DataFrame(validation_rows)
    display(validation_results.round(5))
    print('Frozen before test evaluation:',json.dumps(decisions,indent=2))
    (OUT/'decisions_before_test.json').write_text(json.dumps(decisions,indent=2))
    validation_results.to_json(OUT/'validation_results.json',orient='records',indent=2)

    test_rows=[]
    test_predictions={}
    decade_rows=[]
    for experiment,parts in splits.items():
        train=np.flatnonzero(parts=='train'); test=np.flatnonzero(parts=='test')
        yt=y[test]
        test_predictions[experiment]={'index':test,'y':yt,'scores':{}}
        names=['constant_baseline','logistic_regression',decisions[experiment]['selected_forest']]
        for name in names:
            if name=='constant_baseline':
                scores=np.full(len(test),y[train].mean())
                thresholds=[('default_0.5',.5)]
            else:
                info=fitted[experiment][name]
                scores=info['model'].predict_proba(X.iloc[test])[:,1]
                thresholds=[('default_0.5',.5),('validation_max_f1',info['threshold'])]
            test_predictions[experiment]['scores'][name]=scores
            for rule,t in thresholds:
                pred=scores>=t
                tn,fp,fn,tp=confusion_matrix(yt,pred,labels=[0,1]).ravel()
                test_rows.append({'experiment':experiment,'model':name,'threshold_rule':rule,'threshold':t,
                    'test_rows':len(test),'test_positive_rate':float(yt.mean()),
                    'ap':float(average_precision_score(yt,scores)),'roc_auc':float(roc_auc_score(yt,scores)),
                    'precision':float(precision_score(yt,pred,zero_division=0)),
                    'recall':float(recall_score(yt,pred,zero_division=0)),
                    'f1':float(f1_score(yt,pred,zero_division=0)),
                    'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)})
        winner=decisions[experiment]['selected_model']
        winning=fitted[experiment][winner]
        pred=test_predictions[experiment]['scores'][winner]>=winning['threshold']
        assert np.array_equal(pred, (test_predictions[experiment]['scores'][winner]>=decisions[experiment]['selected_threshold']))
        prediction_table=pd.DataFrame({'id':raw.id.iloc[test].to_numpy(),'high_popularity':yt,
            'score':test_predictions[experiment]['scores'][winner], 'predicted_high_popularity':pred.astype(int)})
        prediction_table.to_csv(OUT/f'{experiment}_test_predictions.csv.gz',index=False,compression='gzip')
        years=pd.to_numeric(raw.release_date.iloc[test].str[:4],errors='coerce').to_numpy()
        decades=(years//10)*10
        for decade in sorted(np.unique(decades)):
            mask=decades==decade; labels=yt[mask]; scores=test_predictions[experiment]['scores'][winner][mask]
            ap=float(average_precision_score(labels,scores)) if 0<labels.sum()<len(labels) else None
            decade_rows.append({'experiment':experiment,'decade':int(decade),'rows':int(mask.sum()),
                'positives':int(labels.sum()),'positive_rate':float(labels.mean()),'ap':ap,
                'precision':float(precision_score(labels,pred[mask],zero_division=0)),
                'recall':float(recall_score(labels,pred[mask],zero_division=0))})
        # Save the actual evaluated training-only fitted model; no refit after test inspection.
        joblib.dump(winning['model'], OUT/f'{experiment}_selected_model.joblib',compress=3)
    test_results=pd.DataFrame(test_rows)
    decade_results=pd.DataFrame(decade_rows)
    display(test_results.round(5))
    display(decade_results.round(5))
    test_results.to_json(OUT/'test_results.json',orient='records',indent=2)
    decade_results.to_json(OUT/'test_by_decade.json',orient='records',indent=2)
    assert np.all(test_results.tp+test_results.fn == np.round(test_results.test_rows*test_results.test_positive_rate))
    assert np.all(test_results[['tn','fp','fn','tp']].sum(axis=1)==test_results.test_rows)
    print('All confusion-matrix totals reconciled.')

    importance_rows=[]
    for experiment,parts in splits.items():
        val=np.flatnonzero(parts=='validation')
        sample,_=train_test_split(val,train_size=min(10000,len(val)-1),stratify=y[val],random_state=SEED)
        model=fitted[experiment][decisions[experiment]['selected_model']]['model']
        sample_x=X.iloc[sample].copy(); sample_y=y[sample]
        base=average_precision_score(sample_y,model.predict_proba(sample_x)[:,1])
        rng=np.random.default_rng(SEED)
        for feature in FEATURES:
            drops=[]
            for repeat in range(3):
                shuffled=sample_x.copy()
                shuffled[feature]=rng.permutation(shuffled[feature].to_numpy())
                drops.append(float(base-average_precision_score(sample_y,model.predict_proba(shuffled)[:,1])))
            importance_rows.append({'experiment':experiment,'feature':feature,'mean_ap_drop':float(np.mean(drops)),
                                    'std_ap_drop':float(np.std(drops)),'validation_sample_rows':len(sample),'sample_base_ap':float(base)})
        print('Permutation interpretation completed:',experiment,flush=True)
    importance_results=pd.DataFrame(importance_rows)
    display(importance_results.sort_values(['experiment','mean_ap_drop'],ascending=[True,False]).round(5))
    importance_results.to_json(OUT/'validation_permutation_importance.json',orient='records',indent=2)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    colors={'logistic_regression':'#3568b0','forest_depth12':'#b35a2b','forest_depth20':'#b35a2b'}
    titles={'version_grouped':'A. Versions kept together','artist_disjoint':'B. Unseen artists'}
    for col,experiment in enumerate(splits):
        info=test_predictions[experiment]; ax=axes[0,col]
        for name,scores in info['scores'].items():
            if name=='constant_baseline': continue
            p,r,_=precision_recall_curve(info['y'],scores)
            ap=average_precision_score(info['y'],scores)
            label='Logistic regression' if name=='logistic_regression' else 'Random forest'
            ax.step(r,p,where='post',color=colors[name],lw=2,label=f'{label}  AP {ap:.3f}')
        ax.axhline(info['y'].mean(),ls='--',color='#777777',label=f'Baseline {info["y"].mean():.3f}')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Recall',ylabel='Precision',title=titles[experiment])
        ax.legend(loc='upper right',frameon=False,fontsize=9)
        ax.grid(alpha=.15)
        imp=importance_results.loc[importance_results.experiment.eq(experiment)].nlargest(8,'mean_ap_drop').sort_values('mean_ap_drop')
        ax=axes[1,col]
        labels=imp.feature.replace({'duration_ms':'duration (ms)'})
        ax.barh(labels,imp.mean_ap_drop,xerr=imp.std_ap_drop,color='#3568b0',alpha=.85,capsize=3)
        ax.axvline(0,color='#777777',lw=.8)
        ax.set(xlabel='Validation AP decrease after shuffling',title='Top features in selected model')
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('Spotify popularity classification | First reproducible experiment',fontsize=16,fontweight='bold')
    fig.savefig(OUT/'model_overview.png',dpi=180,bbox_inches='tight')
    display(fig)
    plt.close(fig)

    metadata={'dataset_sha256':digest.hexdigest(),'seed':SEED,'target_threshold':THRESHOLD,'source_rows':len(raw),
              'versions':versions,'features':FEATURES,'duration_input_unit':'milliseconds, original source values',
              'decisions':decisions,'primary_metric':'average_precision_score, not trapezoidal PR-AUC',
              'probability_calibrated':False,'test_used_for_tuning':False}
    (OUT/'experiment_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Completed. Models are fitted on their respective training sets only; no post-test tuning.')


if __name__ == "__main__":
    main()
