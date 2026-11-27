# Standalone Python version of the corresponding analysis notebook.
# Run via: python run.py stability --data /path/to/archive.zip
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
    import ast, hashlib, json, time, warnings
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.exceptions import ConvergenceWarning

    PRIOR = RESULT_ROOT / 'model_results'
    OUT = RESULT_ROOT / 'stability_results'; OUT.mkdir(exist_ok=True)
    DATA_PATH = Path(os.environ.get("SPOTIFY_DATA_PATH", str(PROJECT_ROOT / "data" / "archive.zip")))
    ORIGINAL_SEED=20261123
    NEW_SEEDS=[20261124,20261125,20261126,20261127]
    MODEL_SEED=20261123
    metadata=json.loads((PRIOR/'experiment_metadata.json').read_text())
    FEATURES=metadata['features']
    AUDIO=FEATURES[:9]
    CONTINUOUS=[c for c in AUDIO if c!='tempo']+['duration_ms']
    with ZipFile(DATA_PATH) as z:
        digest=hashlib.sha256()
        with z.open('tracks.csv') as f:
            for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
        with z.open('tracks.csv') as f: raw=pd.read_csv(f,keep_default_na=False)
    assert digest.hexdigest()==metadata['dataset_sha256']
    manifest=pd.read_csv(PRIOR/'split_manifest.csv.gz')
    assert np.array_equal(manifest.id.to_numpy(),raw.id.to_numpy())
    assert raw.id.is_unique
    groups=manifest.version_group.to_numpy()
    X=raw[FEATURES].copy()
    y=raw.popularity.ge(60).astype('int8').to_numpy()
    @lru_cache(maxsize=None)
    def ids(s):
        values=ast.literal_eval(s)
        assert values and all(isinstance(v,str) and v.strip() for v in values)
        return tuple(sorted(set(values)))
    artists=raw.id_artists.map(ids)
    @lru_cache(maxsize=None)
    def partition(key,namespace,seed):
        h=hashlib.blake2b(f'{seed}|{namespace}|{key}'.encode(),digest_size=8).digest()
        u=int.from_bytes(h,'big')/2**64
        return 'train' if u<.6 else ('validation' if u<.8 else 'test')
    def create_splits(seed):
        a=np.array([partition(g,'version',seed) for g in groups],dtype=object)
        b=[]
        for values in artists:
            parts={partition(v,'artist',seed) for v in values}
            b.append(next(iter(parts)) if len(parts)==1 else 'excluded_cross_artist')
        b=np.array(b,dtype=object)
        keep=np.isin(b,['train','validation','test'])
        overlap=pd.DataFrame({'group':groups[keep],'part':b[keep]}).groupby('group').part.nunique()
        crossing=set(overlap.loc[overlap>1].index)
        b[keep & pd.Series(groups).isin(crossing).to_numpy()]='excluded_cross_version'
        return {'version_grouped':a,'artist_disjoint':b}
    original=create_splits(ORIGINAL_SEED)
    for experiment,parts in original.items():
        assert np.array_equal(parts,manifest[f'{experiment}_split'].to_numpy())
    print('Source hash, record order, and original split reproduction verified.')

    def preprocessing():
        return ColumnTransformer([
            ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),CONTINUOUS),
            ('tempo',Pipeline([('impute',SimpleImputer(missing_values=0,strategy='median',add_indicator=True)),('scale',StandardScaler())]),['tempo']),
            ('categorical',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value=-1)),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),['key','mode']),
            ('time_signature',Pipeline([('impute',SimpleImputer(missing_values=0,strategy='constant',fill_value=-1)),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),['time_signature'])
        ],remainder='drop')
    def models(experiment):
        depth,leaf=(12,20) if experiment=='version_grouped' else (20,5)
        assert metadata['decisions'][experiment]['selected_forest']==f'forest_depth{depth}'
        return {
            'logistic_regression':LogisticRegression(C=1,class_weight='balanced',max_iter=1500,random_state=MODEL_SEED),
            'random_forest':RandomForestClassifier(n_estimators=160,max_depth=depth,min_samples_leaf=leaf,
                max_samples=.65,class_weight='balanced_subsample',max_features='sqrt',n_jobs=int(os.environ.get("SPOTIFY_JOBS", "2")),random_state=MODEL_SEED)
        }
    def best_threshold(labels,scores):
        p,r,t=precision_recall_curve(labels,scores)
        f=np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],out=np.zeros(len(t)),where=(p[:-1]+r[:-1])>0)
        j=np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]
        return float(t[j]),float(f[j])
    def evaluate(seed,experiment,name,rule,threshold,labels,scores):
        pred=scores>=threshold
        tn,fp,fn,tp=confusion_matrix(labels,pred,labels=[0,1]).ravel()
        return {'seed':seed,'experiment':experiment,'model':name,'threshold_rule':rule,'threshold':threshold,
            'test_rows':len(labels),'test_positives':int(labels.sum()),'test_positive_rate':float(labels.mean()),
            'ap':float(average_precision_score(labels,scores)),
            'precision':float(precision_score(labels,pred,zero_division=0)),
            'recall':float(recall_score(labels,pred,zero_division=0)),
            'f1':float(f1_score(labels,pred,zero_division=0)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
    prior=pd.read_json(PRIOR/'test_results.json')
    records=[]
    for _,row in prior.iterrows():
        r=row.to_dict(); r.pop('roc_auc',None)
        r['seed']=ORIGINAL_SEED; r['test_positives']=int(r['tp']+r['fn'])
        if r['model'].startswith('forest_'):r['model']='random_forest'
        records.append(r)
    split_records=pd.read_json(PRIOR/'split_summary.json').assign(seed=ORIGINAL_SEED).to_dict('records')
    overlap_records=pd.read_json(PRIOR/'overlap_checks.json').assign(seed=ORIGINAL_SEED).to_dict('records')
    validation=[]
    print('Original results imported:',len(records),'rows')

    new_manifest=pd.DataFrame({'id':raw.id})
    for seed in NEW_SEEDS:
        print('Starting partition seed',seed,flush=True)
        generated=create_splits(seed)
        for experiment,parts in generated.items():
            new_manifest[f'{experiment}_{seed}']=parts
            idx={p:np.flatnonzero(parts==p) for p in ['train','validation','test']}
            aset={p:{a for values in artists.iloc[rows] for a in values} for p,rows in idx.items()}
            for a,b in combinations(idx,2):
                gv=len(set(groups[idx[a]]) & set(groups[idx[b]]))
                ga=len(aset[a] & aset[b])
                assert gv==0
                if experiment=='artist_disjoint':assert ga==0
                overlap_records.append({'seed':seed,'experiment':experiment,'pair':f'{a}/{b}',
                                         'shared_version_groups':gv,'shared_artists':ga})
            for part in pd.unique(parts):
                rows=np.flatnonzero(parts==part)
                split_records.append({'seed':seed,'experiment':experiment,'partition':part,'rows':len(rows),
                    'positives':int(y[rows].sum()),'positive_rate':float(y[rows].mean()),
                    'multi_artist_rate':float(artists.iloc[rows].map(len).gt(1).mean())})
            train,val,test=idx['train'],idx['validation'],idx['test']
            fitted={}
            for name,estimator in models(experiment).items():
                started=time.monotonic()
                model=Pipeline([('preprocess',preprocessing()),('classifier',estimator)])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    model.fit(X.iloc[train],y[train])
                assert not any(issubclass(w.category,ConvergenceWarning) for w in caught)
                score=model.predict_proba(X.iloc[val])[:,1]
                threshold,vf1=best_threshold(y[val],score)
                fitted[name]=(model,threshold)
                validation.append({'seed':seed,'experiment':experiment,'model':name,'threshold':threshold,
                    'validation_ap':float(average_precision_score(y[val],score)), 'validation_f1':vf1})
                print(seed,experiment,name,'validation AP',round(validation[-1]['validation_ap'],5),
                      'seconds',round(time.monotonic()-started,1),flush=True)
            (OUT/'validation_decisions.json').write_text(json.dumps(validation,indent=2))
            baseline=np.full(len(test),y[train].mean())
            records.append(evaluate(seed,experiment,'constant_baseline','default_0.5',.5,y[test],baseline))
            predictions=pd.DataFrame({'id':raw.id.iloc[test].to_numpy(),'high_popularity':y[test]})
            for name,(model,threshold) in fitted.items():
                scores=model.predict_proba(X.iloc[test])[:,1]
                predictions[name+'_score']=scores
                for rule,t in [('default_0.5',.5),('validation_max_f1',threshold)]:
                    records.append(evaluate(seed,experiment,name,rule,t,y[test],scores))
                print(seed,experiment,name,'test AP',round(records[-1]['ap'],5),flush=True)
            predictions.to_csv(OUT/f'predictions_{experiment}_{seed}.csv.gz',index=False,compression='gzip')
            pd.DataFrame(records).to_json(OUT/'all_test_results.json',orient='records',indent=2)
            del fitted
    new_manifest.to_csv(OUT/'additional_split_manifest.csv.gz',index=False,compression='gzip')
    results=pd.DataFrame(records)
    split_audit=pd.DataFrame(split_records)
    overlap_audit=pd.DataFrame(overlap_records)
    split_audit.to_json(OUT/'all_split_counts.json',orient='records',indent=2)
    overlap_audit.to_json(OUT/'all_overlap_checks.json',orient='records',indent=2)
    assert len(results)==50
    assert (results[['tn','fp','fn','tp']].sum(axis=1)==results.test_rows).all()
    assert (results.tp+results.fn==results.test_positives).all()
    assert overlap_audit.shared_version_groups.eq(0).all()
    assert overlap_audit.loc[overlap_audit.experiment.eq('artist_disjoint'),'shared_artists'].eq(0).all()
    display(results.loc[results.threshold_rule.eq('validation_max_f1')].round(5))

    main=results.loc[results.threshold_rule.eq('validation_max_f1')|results.model.eq('constant_baseline')].copy()
    main['ap_over_baseline']=main.ap/main.test_positive_rate
    summary=main.groupby(['experiment','model'])[['ap','precision','recall','f1','ap_over_baseline']].agg(['mean','std','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index()
    display(summary.round(5))
    summary.to_json(OUT/'stability_summary.json',orient='records',indent=2)
    paired=main.pivot(index=['seed','experiment'],columns='model',values='ap').reset_index()
    paired['forest_minus_logistic']=paired.random_forest-paired.logistic_regression
    paired['forest_minus_baseline']=paired.random_forest-paired.constant_baseline
    display(paired.round(5))
    paired.to_json(OUT/'paired_ap_differences.json',orient='records',indent=2)
    rf_thresholds=results.loc[results.model.eq('random_forest')].pivot(index=['seed','experiment'],columns='threshold_rule',values='f1').reset_index()
    rf_thresholds['tuned_minus_default_f1']=rf_thresholds.validation_max_f1-rf_thresholds['default_0.5']
    display(rf_thresholds.round(5))
    rf_thresholds.to_json(OUT/'forest_threshold_comparison.json',orient='records',indent=2)
    protocol={'source_sha256':digest.hexdigest(),'partition_seeds':[ORIGINAL_SEED]+NEW_SEEDS,
              'model_seed':MODEL_SEED,'target_threshold':60,'new_hyperparameter_search':False,
              'scope':'conditional repeated holdout on one catalogue; not independent external validation',
              'summary':'equal-weight mean, sample standard deviation, minimum and maximum across five partitions',
              'original_model_configurations':metadata['decisions']}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.1),layout='constrained')
    names={'constant_baseline':'Baseline','logistic_regression':'Logistic regression','random_forest':'Random forest'}
    colors={'constant_baseline':'#777777','logistic_regression':'#3568b0','random_forest':'#b35a2b'}
    titles={'version_grouped':'Versions grouped together','artist_disjoint':'Artists kept independent'}
    for ax,experiment in zip(axes,['version_grouped','artist_disjoint']):
        for model in names:
            d=main.loc[main.experiment.eq(experiment)&main.model.eq(model)].sort_values('seed')
            ax.plot(range(1,6),d.ap,marker='o',color=colors[model],lw=1.8,label=names[model])
        ax.set(title=titles[experiment],xlabel='Partition',ylabel='Test average precision',xticks=range(1,6),ylim=(0,.23))
        ax.grid(alpha=.18); ax.legend(loc='upper right',frameon=False,fontsize=9)
    fig.savefig(OUT/'stability_overview.png',dpi=200,bbox_inches='tight')
    display(fig);plt.close(fig)
    print('Completed all five partitions. No model changes were made from test outcomes.')

    training=raw.loc[manifest.version_grouped_split.eq('train')].copy()
    training['high_popularity']=training.popularity.ge(60)
    descriptive=[]
    for feature in ['loudness','acousticness','energy','danceability','valence','tempo']:
        descriptive.append({'feature':feature,
            'ordinary_median':float(training.loc[~training.high_popularity,feature].median()),
            'popular_median':float(training.loc[training.high_popularity,feature].median())})
    descriptive_table=pd.DataFrame(descriptive)
    display(descriptive_table)
    descriptive_table.to_json(OUT/'training_descriptive_stats.json',orient='records',indent=2)


if __name__ == "__main__":
    main()
