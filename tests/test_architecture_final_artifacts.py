import json
import pathlib
import tempfile
import unittest
import hashlib
import contextlib
import pandas as pd
from types import SimpleNamespace
from unittest.mock import patch


class FinalArtifactTests(unittest.TestCase):
    def test_six_model_final_recovers_completed_inference_and_resumes_without_rerun(self):
        from gated_dual_ema_msd.operations import architecture_final as module
        from gated_dual_ema_msd.config.contracts import MODEL_REVISION,DATASET_REVISIONS
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp); repo=root/'repo';data=repo/'data/external/vinli/test.jsonl'
            data.parent.mkdir(parents=True)
            rows=[dict(id=str(i),label=label) for i,label in enumerate(('E','C','N'))]
            data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            fingerprints=dict(splits={'train':{'sha256':'train'},'dev':{'sha256':'dev'},'test':{'sha256':module.sha256_file(data)}})
            checkpoint=root/'weights.bin';checkpoint.write_bytes(b'weights')
            sources={}
            for seed in (42,2024,3407):
                for method in ('M3_FULL','ARCH_REL256'):
                    sources[method,seed]=dict(dataset='vinli',experiment_id=method,seed=seed,
                        hf_repo_id=f'test/{method}-{seed}',hf_revision='a'*40,hf_checkpoint_path='stage2_checkpoint/pytorch_model.bin',
                        hf_checkpoint_sha256=module.sha256_file(checkpoint),git_sha='c'*40,model_revision=MODEL_REVISION,
                        batch_manifest_sha256='d'*64,selected_weight_source='ema',data_fingerprints=fingerprints)
            remote={}; calls=[]; interrupted=[False]
            class API:
                def repo_info(self,**kwargs): return SimpleNamespace(sha='b'*40,private=False)
                def list_repo_files(self,**kwargs): return [module.FINAL_LEDGER,module.FINAL_PREDICTIONS] if kwargs['repo_id'] in remote else []
            def download(**kwargs):
                name=kwargs['filename'];destination=pathlib.Path(kwargs['local_dir'])/name;destination.parent.mkdir(parents=True,exist_ok=True)
                if name in (module.FINAL_LEDGER,module.FINAL_PREDICTIONS): destination.write_bytes(remote[kwargs['repo_id']][name])
                elif name=='run_metadata.json':
                    source=next(s for s in sources.values() if s['hf_repo_id']==kwargs['repo_id'])
                    metadata={**source,'checkpoint_manifest':{'pytorch_model.bin':{'sha256':module.sha256_file(checkpoint)}}}
                    destination.write_text(json.dumps(metadata))
                else: destination.write_bytes(checkpoint.read_bytes())
                return str(destination)
            def inference(args):
                method=args[args.index('--experiment_id')+1];calls.append(method)
                pred=pathlib.Path(args[args.index('--output_csv')+1]);ledger=pathlib.Path(args[args.index('--output_json')+1])
                frame=pd.DataFrame([dict(sample_id=str(i),gold_label=label,pred_label=label,
                    logit_E=int(i==0),logit_C=int(i==1),logit_N=int(i==2)) for i,label in enumerate(('E','C','N'))])
                frame.to_csv(pred,index=False)
                ledger.write_text(json.dumps(dict(state='complete',dataset='vinli',split='test',experiment_id=method,
                    checkpoint_sha256=module.sha256_file(checkpoint),model_revision=MODEL_REVISION,
                    data_revision=DATASET_REVISIONS['vinli'],max_length=512,eval_precision='fp32',test_evaluations=1,
                    prediction_sha256=module.sha256_file(pred),row_count=3,metrics={'macro_f1':1.,'accuracy':1.})))
                if not interrupted[0]:
                    interrupted[0]=True
                    raise KeyboardInterrupt('VM loss after inference completion')
            def publish(repo_id,pred,ledger):
                remote[repo_id]={module.FINAL_LEDGER:ledger.read_bytes(),module.FINAL_PREDICTIONS:pred.read_bytes()}
                return 'b'*40
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(module,'confirmation_decision',return_value={'passes_confirmation':True}))
                stack.enter_context(patch.object(module,'verified_result',side_effect=lambda root,method,seed:sources[method,seed]))
                stack.enter_context(patch.object(module,'prepare_data',return_value={'vinli':fingerprints}))
                stack.enter_context(patch.object(module,'_api',return_value=API()))
                stack.enter_context(patch.object(module,'hf_hub_download',side_effect=download))
                stack.enter_context(patch.object(module,'publish_final_evaluation',side_effect=publish))
                stack.enter_context(patch.object(module,'environment_metadata',return_value={}))
                stack.enter_context(patch.object(module,'_linked_manifests',return_value={'environment_contract':module.environment_contract({})}))
                stack.enter_context(patch.object(module.torch.cuda,'is_available',return_value=True))
                stack.enter_context(patch.object(module.torch.cuda,'empty_cache'))
                stack.enter_context(patch.object(module.evaluate,'main',side_effect=inference))
                params=dict(screening_root=root/'screen',confirmation_root=root/'confirm',candidate='ARCH_REL256',repo_dir=repo,output_root=root/'final')
                with self.assertRaises(KeyboardInterrupt): module.evaluate_final_batch(**params)
                result=module.evaluate_final_batch(**params)
                self.assertEqual(result['verified_evaluations'],6)
                self.assertEqual(len(calls),6,'Completed first inference must be recovered without rerunning test')
                module.evaluate_final_batch(**params)
                params['output_root']=root/'fresh-drive'
                module.evaluate_final_batch(**params)
                self.assertEqual(len(calls),6,'Local and HF resume must both skip all inference')
                self.assertEqual(len(remote),6)

    def test_final_upload_preserves_screening_files_and_reads_back_hashes(self):
        from gated_dual_ema_msd.operations import architecture_final as module
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            pred=root/'test_predictions.csv';pred.write_text('sample_id,gold_label,pred_label\nx,E,E\n')
            ledger=root/'evaluation.json';ledger.write_text(json.dumps(dict(state='complete',test_evaluations=1)))
            operations=[]
            class API:
                def repo_info(self,**kwargs): return SimpleNamespace(sha='a'*40)
                def list_repo_files(self,**kwargs): return []
                def create_commit(self,**kwargs):
                    operations.extend(kwargs['operations']);return SimpleNamespace(oid='b'*40)
            def download(**kwargs):
                self.assertEqual(kwargs['revision'],'b'*40)
                return str(ledger if kwargs['filename'].endswith('evaluation.json') else pred)
            with patch.object(module,'_api',return_value=API()),patch.object(module,'hf_hub_download',side_effect=download):
                revision=module.publish_final_evaluation('test/repo',pred,ledger)
            self.assertEqual(revision,'b'*40)
            self.assertEqual({op.path_in_repo for op in operations},
                             {'final_evaluation/evaluation.json','final_evaluation/test_predictions.csv'})


if __name__ == '__main__': unittest.main()
