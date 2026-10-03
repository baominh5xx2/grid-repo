"""Test-aware architecture ranking is explicit and tied to saved peak evidence."""
import json
import pathlib
import tempfile
import runpy
import copy
import shutil
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


class ArchitectureTestPeakTests(unittest.TestCase):
    def prepare(self, root, perfect_method='ARCH_ALIGN256'):
        from gated_dual_ema_msd.operations.notebook_batch import manifest_digest
        from gated_dual_ema_msd.operations.architecture_search import SCREENING_METHODS
        from gated_dual_ema_msd.training.direct import compute_metrics
        from gated_dual_ema_msd.tracking.hf import sha256_file
        manifest=dict(git_sha='a'*40,config={'test_peak_exploratory':True,'frozen_final':False},
                      test_locked=False,protocol='vinli_architecture_test_peak_exploratory',
                      selection_policy='exploratory_test_macro_f1',model_name='uitnlp/CafeBERT',model_revision='b'*40,
                      precision='bf16',evaluation_precision='fp32',label_order=['E','C','N'],datasets=['vinli'],
                      data_fingerprints={'vinli':{}},environment_contract={'gpu':'test'})
        (root/'batch_manifest.json').write_text(json.dumps(manifest))
        for method in SCREENING_METHODS:
            pred=[0,1,2] if method==perfect_method else [0,1,0]
            metrics=compute_metrics(pred,[0,1,2]);folder=root/'vinli'/method/'seed42';folder.mkdir(parents=True)
            frame=pd.DataFrame([dict(sample_id=str(i),gold_label=g,pred_label=('E','C','N')[p],
                logit_E=int(p==0),logit_C=int(p==1),logit_N=int(p==2)) for i,(g,p) in enumerate(zip(('E','C','N'),pred))])
            frame.to_csv(folder/'test_predictions_peak.csv',index=False)
            pd.DataFrame([dict(optimizer_step=120,epoch=1,test_macro_f1=metrics['macro_f1']-.01,weight_source='ema'),
                          dict(optimizer_step=150,epoch=2,test_macro_f1=metrics['macro_f1'],weight_source='ema')]).to_csv(folder/'test_curve.csv',index=False)
            result=dict(dataset='vinli',experiment_id=method,seed=42,hparams={'max_length':512,'train_precision':'bf16','fp32_eval':True},
                selection_policy='dev_macro_f1',selected_weight_source='ema',test_peak_exploratory=True,
                test_selection_policy='exploratory_test_macro_f1',test_evaluations=3,test={'macro_f1':.5},
                final_dev={'macro_f1':.99 if method=='M3_FULL' else .50},peak_test_metrics=metrics,
                peak_test_macro_f1=metrics['macro_f1'],peak_test_step=150,peak_test_checkpoint='best_test_model.pt',
                hf_revision='b'*40,hf_repo_id='test/'+method,hf_exploratory_peak_checkpoint_path='stage2_checkpoint/exploratory_best_test_model.pt',
                artifact_readback_verified=True,batch_manifest_sha256=manifest_digest(manifest),git_sha='a'*40)
            result.update(peak_test_epoch=2,peak_test_weight_source='ema',peak_test_dev_metrics={'macro_f1':.50},
                          hf_exploratory_peak_checkpoint_sha256='c'*64)
            (folder/'test_peak.json').write_text(json.dumps(dict(optimizer_step=150,epoch=2,weight_source='ema',
                metrics=metrics,dev_metrics_at_peak=result['peak_test_dev_metrics'],selection_policy='exploratory_test_macro_f1')))
            result.update(test_curve_sha256=sha256_file(folder/'test_curve.csv'),peak_metadata_sha256=sha256_file(folder/'test_peak.json'),
                          peak_prediction_sha256=sha256_file(folder/'test_predictions_peak.csv'))
            (folder/'verified_run.json').write_text(json.dumps(result))

    def test_peak_metadata_must_match_saved_step_epoch_and_weight_source(self):
        from gated_dual_ema_msd.operations.architecture_test_peak import select_peak_candidate
        from gated_dual_ema_msd.tracking.hf import sha256_file
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);self.prepare(root)
            folder=root/'vinli/ARCH_ALIGN256/seed42';meta=folder/'test_peak.json';marker=folder/'verified_run.json'
            metadata=json.loads(meta.read_text());metadata['epoch']=3;meta.write_text(json.dumps(metadata))
            result=json.loads(marker.read_text());result['peak_metadata_sha256']=sha256_file(meta);marker.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'metadata'): select_peak_candidate(root)

    def test_notebook_binds_test_and_launches_only_peak_winner_for_extra_seeds(self):
        from gated_dual_ema_msd.cli.matrix import MatrixJob
        builder=runpy.run_path(str(ROOT/'scripts/build_vinli_architecture_notebook.py'))
        cells={c['metadata']['tags'][0]:''.join(c['source']) for c in builder['build_notebook']()['cells'] if c['cell_type']=='code'}
        namespace={};exec(cells['configuration'],namespace)
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);calls=[]
            fingerprint={'vinli':{'revision':namespace['PINNED_VINLI_REVISION'],'splits':{s:{'row_count':n} for s,n in namespace['EXPECTED_SPLIT_SIZES'].items()}}}
            def prepare(repo,datasets,*,frozen_final):
                self.assertTrue(frozen_final);return fingerprint
            namespace.update(REPO_DIR=root/'repo',OUTPUT_ROOT=root/'screen',LOCAL_RUN_ROOT=root/'work',SOURCE_SHA='a'*40,
                MODEL_NAME=namespace['PINNED_MODEL_NAME'],MODEL_REVISION=namespace['PINNED_MODEL_REVISION'],
                CONFIG={**namespace['HYPERPARAMS'],'test_peak_exploratory':True,'frozen_final':False},
                JOBS=[MatrixJob(**s) for s in namespace['JOB_SPECS']],prepare_data=prepare,environment_metadata=lambda:{},
                bind_manifest=lambda path,value:'signature',write_json=lambda *args:None,
                verify_hf_write_access=lambda *args:None,run_batch=lambda *args:calls.append(args))
            exec(cells['data-preparation'],namespace)
            self.assertFalse(namespace['BATCH_MANIFEST']['test_locked'])
            self.assertEqual(namespace['BATCH_MANIFEST']['selection_policy'],'exploratory_test_macro_f1')
            self.assertIn('test',namespace['BATCH_MANIFEST']['data_fingerprints']['vinli']['splits'])
            jobs=[MatrixJob('vinli',m,s) for s in (2024,3407) for m in ('M3_FULL','ARCH_ALIGN256')]
            with patch('gated_dual_ema_msd.operations.architecture_test_peak.peak_confirmation_jobs',return_value=jobs):
                exec(cells['confirmation'],namespace)
                self.assertEqual(calls,[])
                with self.assertRaisesRegex(ValueError,'authorize=True'): namespace['launch_confirmation']('ARCH_ALIGN256')
                namespace['launch_confirmation']('ARCH_ALIGN256',authorize=True)
            self.assertEqual(len(calls),1)
            config=calls[0][-2];manifest=calls[0][-1]
            self.assertTrue(config['test_peak_exploratory']);self.assertFalse(config['frozen_final'])
            self.assertEqual(config['patience'],0)
            self.assertEqual(manifest['protocol'],'vinli_architecture_test_peak_robustness')
            self.assertEqual(manifest['methods'],['M3_FULL','ARCH_ALIGN256'])
            for job in jobs:
                slug=f"{config['hf_prefix']}-{job.dataset}-{job.experiment_id.lower().replace('_','-')}-seed{job.seed}"
                self.assertLessEqual(len(slug),96)

    def test_robustness_reports_all_peaks_and_rejects_foreign_study(self):
        from gated_dual_ema_msd.operations.architecture_test_peak import peak_robustness_summary
        from gated_dual_ema_msd.operations.notebook_batch import manifest_digest
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);screen=root/'screen';paired=root/'paired';screen.mkdir();paired.mkdir()
            self.prepare(screen)
            pilot=json.loads((screen/'batch_manifest.json').read_text())
            manifest={**copy.deepcopy(pilot),'protocol':'vinli_architecture_test_peak_robustness',
                'screening_manifest_sha256':manifest_digest(pilot),'selected_candidate':'ARCH_ALIGN256',
                'methods':['M3_FULL','ARCH_ALIGN256'],'seeds':[2024,3407],
                'jobs':[dict(dataset='vinli',experiment_id=m,seed=s) for s in (2024,3407) for m in ('M3_FULL','ARCH_ALIGN256')]}
            (paired/'batch_manifest.json').write_text(json.dumps(manifest))
            for seed in (2024,3407):
                for method in ('M3_FULL','ARCH_ALIGN256'):
                    folder=paired/'vinli'/method/f'seed{seed}'
                    shutil.copytree(screen/'vinli'/method/'seed42',folder)
                    marker=folder/'verified_run.json';result=json.loads(marker.read_text())
                    result.update(seed=seed,batch_manifest_sha256=manifest_digest(manifest));marker.write_text(json.dumps(result))
            runs,summary=peak_robustness_summary(screen,paired,'ARCH_ALIGN256')
            self.assertEqual(len(runs),6);self.assertEqual(summary.seeds.tolist(),[3,3])
            self.assertEqual(set(runs.selection_policy),{'exploratory_test_macro_f1'})
            manifest['environment_contract']={'gpu':'different'};(paired/'batch_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'provenance'): peak_robustness_summary(screen,paired,'ARCH_ALIGN256')

    def test_exact_tie_retains_control_without_duplicate_robustness_jobs(self):
        from gated_dual_ema_msd.operations.architecture_test_peak import select_peak_candidate,peak_confirmation_jobs,peak_capacity_control_jobs
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);self.prepare(root,perfect_method=None)
            self.assertEqual(select_peak_candidate(root),'M3_FULL')
            jobs=peak_confirmation_jobs(root,'M3_FULL')
            self.assertEqual([(j.experiment_id,j.seed) for j in jobs],[('M3_FULL',2024),('M3_FULL',3407)])
            self.assertEqual(peak_capacity_control_jobs(root),[])

    def test_default_notebook_scans_test_and_does_not_stop_on_dev(self):
        builder=runpy.run_path(str(ROOT/'scripts/build_vinli_architecture_notebook.py'))
        notebook=builder['build_notebook']()
        cells={c['metadata']['tags'][0]:''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code'}
        namespace={};exec(cells['configuration'],namespace)
        self.assertTrue(namespace['TEST_PEAK_EXPLORATORY'])
        self.assertFalse(namespace['FROZEN_FINAL'])
        self.assertEqual(namespace['HYPERPARAMS']['patience'],0)
        self.assertEqual(namespace['HYPERPARAMS']['eval_steps'],30)
        self.assertEqual(namespace['EXPECTED_SPLIT_SIZES'],{'train':18282,'dev':2255,'test':2264})
        self.assertNotEqual(namespace['RUN_GROUP'],'vinli-architecture-bf16-seed42-2026-10-03')
        self.assertIn('test_peak_table',cells['summary'])
        self.assertNotIn('select_candidate',cells['summary'])
        self.assertNotIn('run_final_test', '\n'.join(cells.values()))
        for c in notebook['cells']:
            if c['cell_type']=='code':
                self.assertEqual(c['outputs'],[])
                compile(''.join(c['source']),c['id'],'exec')

    def test_peak_ranking_ignores_dev_gate_and_rejects_inconsistent_evidence(self):
        from gated_dual_ema_msd.operations.architecture_test_peak import select_peak_candidate,test_peak_table,peak_confirmation_jobs
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);self.prepare(root)
            self.assertEqual(select_peak_candidate(root),'ARCH_ALIGN256')
            table=test_peak_table(root)
            self.assertEqual(table.iloc[0].experiment_id,'ARCH_ALIGN256')
            self.assertEqual(table.iloc[0].peak_test_step,150)
            jobs=peak_confirmation_jobs(root,'ARCH_ALIGN256')
            self.assertEqual([(j.experiment_id,j.seed) for j in jobs],[(m,s) for s in (2024,3407) for m in ('M3_FULL','ARCH_ALIGN256')])
            with self.assertRaises(ValueError): peak_confirmation_jobs(root,'ARCH_REL256')
            curve=root/'vinli/ARCH_ALIGN256/seed42/test_curve.csv'
            frame=pd.read_csv(curve);frame.loc[0,'test_macro_f1']=1.01;frame.to_csv(curve,index=False)
            with self.assertRaisesRegex(ValueError,'curve|peak'): select_peak_candidate(root)

    def test_missing_and_dev_only_runs_do_not_count_as_peak_results(self):
        from gated_dual_ema_msd.operations.architecture_test_peak import select_peak_candidate
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);self.prepare(root)
            marker=root/'vinli/ARCH_REL256/seed42/verified_run.json'
            result=json.loads(marker.read_text());result['test_peak_exploratory']=False;marker.write_text(json.dumps(result))
            with self.assertRaises(ValueError): select_peak_candidate(root)


if __name__=='__main__': unittest.main()
