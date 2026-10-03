import json
import copy
import pathlib
import tempfile
import unittest


class ArchitectureDecisionTests(unittest.TestCase):
    def test_capacity_control_matches_selected_architecture(self):
        from unittest.mock import patch
        from gated_dual_ema_msd.operations.architecture_search import capacity_control_jobs
        for candidate, control in [('ARCH_ALIGN256','ARCH_REL304'),
                                   ('ARCH_CONDPOOL128','ARCH_REL230'),('ARCH_REL256','ARCH_REL512')]:
            with patch('gated_dual_ema_msd.operations.architecture_search.select_candidate',return_value=candidate):
                jobs=capacity_control_jobs(pathlib.Path('unused'))
            self.assertEqual([(j.dataset,j.experiment_id,j.seed) for j in jobs],[('vinli',control,42)])

    def test_confirmation_rejects_foreign_manifest_before_dev_comparison(self):
        from unittest.mock import patch
        from gated_dual_ema_msd.operations.architecture_search import confirmation_decision
        from gated_dual_ema_msd.operations.notebook_batch import manifest_digest
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);screen=root/'screen';confirm=root/'confirm'
            screen.mkdir();confirm.mkdir()
            pilot=dict(git_sha='a'*40,model_name='uitnlp/CafeBERT',model_revision='b'*40,
                       precision='bf16',evaluation_precision='fp32',test_locked=True,
                       label_order=['E','C','N'],datasets=['vinli'],data_fingerprints={'vinli':{}},
                       environment_contract={'gpu':'test'},config={'lr':1e-5,'hf_prefix':'pilot'},
                       selection_policy='best_active_ema_dev_macro_f1')
            paired={**copy.deepcopy(pilot),'screening_manifest_sha256':manifest_digest(pilot),
                    'protocol':'vinli_architecture_confirmation_train_dev','selected_candidate':'ARCH_REL256',
                    'methods':['M3_FULL','ARCH_REL256'],'seeds':[2024,3407],
                    'jobs':[dict(dataset='vinli',experiment_id=m,seed=s) for s in (2024,3407) for m in ('M3_FULL','ARCH_REL256')]}
            paired['config']['hf_prefix']='confirm'
            (screen/'batch_manifest.json').write_text(json.dumps(pilot))
            pairs=dict(final_dev={'macro_f1':.834})
            for mutation in ('screening_manifest_sha256','git_sha','model_revision','data_fingerprints','environment_contract','config'):
                changed=copy.deepcopy(paired)
                changed[mutation] = {'bad':True} if isinstance(changed[mutation],dict) else 'foreign'
                (confirm/'batch_manifest.json').write_text(json.dumps(changed))
                with self.subTest(mutation=mutation), patch('gated_dual_ema_msd.operations.architecture_search.confirmation_jobs'), patch('gated_dual_ema_msd.operations.architecture_search.verified_result') as reader:
                    with self.assertRaisesRegex(ValueError,'provenance|manifest|recipe'):
                        confirmation_decision(screen,confirm,'ARCH_REL256')
                    reader.assert_not_called()
            (confirm/'batch_manifest.json').write_text(json.dumps(paired))
            with patch('gated_dual_ema_msd.operations.architecture_search.confirmation_jobs'), patch('gated_dual_ema_msd.operations.architecture_search.verified_result',return_value=pairs):
                result=confirmation_decision(screen,confirm,'ARCH_REL256')
            self.assertTrue(result['complete'])
            self.assertFalse(result['passes_confirmation'])

    def prepare(self, root, missing=None, slow=False, e_drop=False):
        from gated_dual_ema_msd.operations.architecture_search import SCREENING_METHODS
        for method in SCREENING_METHODS:
            if method == missing:
                continue
            metrics = dict(macro_f1=.83 if method == 'M3_FULL' else .834,
                           f1_E=.83, f1_C=.83 if method == 'M3_FULL' else .834,
                           f1_N=.83 if method == 'M3_FULL' else .834)
            if e_drop and method != 'M3_FULL': metrics['f1_E'] -= .01
            value = dict(dataset='vinli',experiment_id=method,seed=42,final_dev=metrics,
                         hf_revision='a'*40, artifact_readback_verified=True,
                         hf_repo_id='test/'+method,selection_policy='dev_macro_f1',
                         selected_weight_source='ema',test=None,test_evaluations=0,
                         hparams=dict(max_length=512,train_precision='bf16',fp32_eval=True,
                                      additional_head_parameters=2761859),
                         train_seconds_per_optimizer_step=2 if slow and method != 'M3_FULL' else 1)
            folder=root/'vinli'/method/'seed42';folder.mkdir(parents=True)
            (folder/'verified_run.json').write_text(json.dumps(value))

    def test_confirmation_requires_complete_verified_screen_and_dev_signal(self):
        from gated_dual_ema_msd.operations.architecture_search import select_candidate, confirmation_jobs
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);self.prepare(root)
            candidate=select_candidate(root)
            jobs=confirmation_jobs(root,candidate)
            self.assertEqual({j.seed for j in jobs},{2024,3407})
            self.assertEqual(len(jobs),4)
            marker=root/'vinli'/candidate/'seed42'/'verified_run.json'
            value=json.loads(marker.read_text());value['test_evaluations']=2;marker.write_text(json.dumps(value))
            with self.assertRaises(ValueError): select_candidate(root)

    def test_missing_or_slow_or_class_tradeoff_cannot_auto_advance(self):
        from gated_dual_ema_msd.operations.architecture_search import select_candidate, SCREENING_METHODS
        for params in [dict(missing=SCREENING_METHODS[-1]),dict(slow=True),dict(e_drop=True)]:
            with self.subTest(params=params), tempfile.TemporaryDirectory() as tmp:
                root=pathlib.Path(tmp);self.prepare(root,**params)
                with self.assertRaises(ValueError): select_candidate(root)


if __name__ == '__main__': unittest.main()
