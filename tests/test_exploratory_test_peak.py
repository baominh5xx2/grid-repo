"""Keep test-aware checkpoint exploration explicit and separate from dev selection."""
import json
import pathlib
import tempfile
import hashlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gated_dual_ema_msd.cli import train
from gated_dual_ema_msd.training import direct
from verify_source import TinyBackbone, TinyTokenizer, rows


class ExploratoryTestPeakTests(unittest.TestCase):
    def test_full_budget_retains_late_peak_weights_and_earliest_tie_despite_dev_decline(self):
        import torch
        actual_evaluate=direct.evaluate_model
        scans=[];dev_calls=0
        def evaluate(model,evaluation_rows,*args,**kwargs):
            nonlocal dev_calls
            metrics,frame=actual_evaluate(model,evaluation_rows,*args,**kwargs)
            if evaluation_rows[0]['id'].startswith('test-'):
                scores=[.60,.65,.80,.82,.70,.82,.10]
                metrics['macro_f1']=scores[len(scans)]
                scans.append({k:v.detach().clone() for k,v in model.state_dict().items()})
            else:
                dev_calls+=1;metrics['macro_f1']=.95-.01*dev_calls
            return metrics,frame
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(train,'load_nli_dataset',return_value=({'train':rows('train',6),'dev':rows('dev',3),'test':rows('test',3)},TinyTokenizer())), patch('transformers.AutoModel.from_pretrained',side_effect=lambda *args,**kwargs:TinyBackbone()), patch.object(direct,'evaluate_model',side_effect=evaluate):
                train.main(['--experiment_id','M3_FULL','--dataset','vianli','--epochs','2',
                    '--physical_batch_size','2','--grad_accum','1','--eval_steps','1','--patience','0',
                    '--ema_start_step','1','--test_peak_exploratory','--no_wandb','--output_dir',directory])
            folder=pathlib.Path(directory)/'vianli/M3_FULL/seed42';result=json.loads((folder/'result.json').read_text())
            self.assertEqual(result['optimizer_steps'],6)
            self.assertEqual(result['best_dev_step'],1)
            self.assertEqual(result['peak_test_step'],4)
            self.assertEqual(result['peak_test_epoch'],2)
            self.assertEqual(result['peak_test_weight_source'],'ema')
            self.assertAlmostEqual(result['peak_test_dev_metrics']['macro_f1'],.87)
            state=torch.load(folder/'best_test_model.pt',weights_only=True)
            for key,value in scans[3].items(): self.assertTrue(torch.equal(state[key],value),key)
            metadata=json.loads((folder/'test_peak.json').read_text())
            self.assertEqual(metadata['optimizer_step'],4)
            self.assertEqual(metadata['selection_policy'],'exploratory_test_macro_f1')
            curve=__import__('pandas').read_csv(folder/'test_curve.csv')
            self.assertEqual(len(curve),6)
            self.assertEqual(curve.loc[curve.is_test_peak,'optimizer_step'].tolist(),[1,2,3,4])
            self.assertTrue({'test_f1_E','test_f1_C','test_f1_N'}.issubset(curve.columns))

    def test_peak_checkpoint_and_curve_are_separate_from_dev_selected_result(self):
        actual_evaluate = direct.evaluate_model
        test_calls = 0
        def evaluate(model, evaluation_rows, *args, **kwargs):
            nonlocal test_calls
            metrics, frame = actual_evaluate(model, evaluation_rows, *args, **kwargs)
            if evaluation_rows[0]["id"].startswith("test-"):
                test_calls += 1
                metrics["macro_f1"] = 0.9 if test_calls == 1 else 0.1
            return metrics, frame
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(train, "load_nli_dataset", return_value=({"train": rows("train", 8), "dev": rows("dev", 3), "test": rows("test", 3)}, TinyTokenizer())), patch("transformers.AutoModel.from_pretrained", side_effect=lambda *args, **kwargs: TinyBackbone()), patch.object(direct, "evaluate_model", side_effect=evaluate):
                train.main(["--experiment_id", "M3_FULL", "--dataset", "vianli", "--epochs", "1",
                            "--physical_batch_size", "4", "--grad_accum", "1", "--eval_steps", "1",
                            "--ema_start_step", "1", "--test_peak_exploratory", "--no_wandb", "--output_dir", directory])
            run_dir = pathlib.Path(directory) / "vianli/M3_FULL/seed42"
            result = json.loads((run_dir / "result.json").read_text())
            self.assertEqual(test_calls, 3)
            self.assertEqual(result["test_evaluations"], 3)
            self.assertEqual(result["selection_policy"], "dev_macro_f1")
            self.assertEqual(result["test_selection_policy"], "exploratory_test_macro_f1")
            self.assertTrue(result["test_peak_exploratory"])
            self.assertEqual(result["peak_test_macro_f1"], 0.9)
            self.assertEqual(result["peak_test_step"], 1)
            self.assertEqual(result["test"]["macro_f1"], 0.1)
            self.assertTrue((run_dir / "best_test_model.pt").is_file())
            self.assertTrue((run_dir / "test_predictions_peak.csv").is_file())
            self.assertEqual(len(__import__('pandas').read_csv(run_dir / "test_curve.csv")), 2)

    def test_cli_modes_cannot_be_combined(self):
        with self.assertRaises(ValueError):
            train.main(["--frozen_final", "--test_peak_exploratory"])
        with self.assertRaises(ValueError):
            train.main(["--no_test", "--test_peak_exploratory"])

    def test_publication_includes_both_weights_curve_and_all_predictions(self):
        from gated_dual_ema_msd.operations import notebook_batch as batch
        from gated_dual_ema_msd.cli.matrix import MatrixJob, write_summaries
        import pandas as pd
        job = MatrixJob("vianli", "M3_FULL", 42)
        config = dict(test_peak_exploratory=True, frozen_final=False, hf_namespace="offline",
                      hf_prefix="synthetic", hf_private=False)
        result = dict(dataset="vianli", experiment_id="M3_FULL", seed=42,
                      hparams=dict(train_precision="bf16", fp32_eval=True, max_length=512, model_configuration={}),
                      selection_policy="dev_macro_f1", test_selection_policy="exploratory_test_macro_f1",
                      test_peak_exploratory=True, test_evaluations=2, peak_test_step=100,
                      peak_test_macro_f1=1.0, peak_test_checkpoint="synthetic/best_test_model.pt",
                      selected_weight_source="ema", final_dev=dict(macro_f1=1.0, accuracy=1.0),
                      test=dict(macro_f1=1.0, accuracy=1.0), wandb_run_path=None)
        tracked = SimpleNamespace(state="finished", summary={})
        wandb = SimpleNamespace(Api=lambda: SimpleNamespace(run=lambda path: tracked))
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run_dir = job.output_dir(root / "results")
            run_dir.mkdir(parents=True)
            batch.write_json(run_dir / "result.json", result)
            (run_dir / "best_model.pt").write_bytes(b"synthetic-dev-weights")
            (run_dir / "best_test_model.pt").write_bytes(b"synthetic-test-peak-weights")
            (run_dir / "dev_history.csv").write_text("optimizer_step,current_dev_macro_f1,ema_dev_macro_f1\n100,0.9,1.0\n")
            (run_dir / "architecture_diagnostics.json").write_text('{"schema_version":1}')
            (run_dir / "test_peak.json").write_text('{"optimizer_step":100,"selection_policy":"exploratory_test_macro_f1"}')
            splits = {}
            for split in ("dev", "test"):
                data_path = root / batch.DATA_DIRS[job.dataset] / f"{split}.jsonl"
                data_path.parent.mkdir(parents=True, exist_ok=True)
                data_path.write_text(json.dumps(dict(id=split+'-1', premise='p', hypothesis='h', label='E'))+'\n')
                splits[split] = dict(sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(), row_count=1)
                frame = pd.DataFrame([dict(sample_id=split+'-1', gold_label='E', pred_label='E', logit_E=1.0, logit_C=0.0, logit_N=0.0)])
                frame.to_csv(run_dir / f"{split}_predictions.csv", index=False)
                if split == "dev":
                    frame.to_csv(run_dir / "dev_predictions_current.csv", index=False)
                if split == "test":
                    frame.to_csv(run_dir / "test_predictions_peak.csv", index=False)
            pd.DataFrame([dict(optimizer_step=100, test_macro_f1=1.0)]).to_csv(run_dir / "test_curve.csv", index=False)
            manifest = dict(git_sha="a"*40, data_fingerprints={job.dataset: dict(splits=splits)}, protocol="exploratory_m3_bf16_test_peak")
            saver = SimpleNamespace(save_pretrained=lambda path: (pathlib.Path(path)/'config.json').write_text('{}'))
            def push(hf_config, checkpoint, predictions, metadata):
                self.assertEqual((checkpoint/'pytorch_model.bin').read_bytes(), b"synthetic-dev-weights")
                self.assertEqual((checkpoint/'exploratory_best_test_model.pt').read_bytes(), b"synthetic-test-peak-weights")
                self.assertTrue((checkpoint/'exploratory_test_curve.csv').exists())
                self.assertEqual((checkpoint/'dev_history.csv').read_bytes(), (run_dir/'dev_history.csv').read_bytes())
                self.assertEqual((checkpoint/'architecture_diagnostics.json').read_bytes(), (run_dir/'architecture_diagnostics.json').read_bytes())
                self.assertEqual((checkpoint/'test_peak.json').read_bytes(), (run_dir/'test_peak.json').read_bytes())
                self.assertEqual(set(predictions), {'vianli_dev', 'vianli_dev_current', 'vianli_test', 'vianli_test_exploratory_peak'})
                self.assertTrue(metadata['test_peak_exploratory'])
                self.assertTrue(metadata['target_test_accessed'])
                return 'offline/synthetic', 'b'*40
            with patch.dict(sys.modules, {'wandb': None}), patch.object(batch.AutoTokenizer, 'from_pretrained', return_value=saver), patch.object(batch.AutoConfig, 'from_pretrained', return_value=saver), patch.object(batch, 'push_run_artifacts', side_effect=push):
                published = batch.publish_run(run_dir, root, job, config, manifest)
            self.assertTrue(published['artifact_readback_verified'])
            self.assertEqual(published['test_curve_sha256'],hashlib.sha256((run_dir/'test_curve.csv').read_bytes()).hexdigest())
            self.assertEqual(published['hf_exploratory_peak_checkpoint_sha256'],hashlib.sha256((run_dir/'best_test_model.pt').read_bytes()).hexdigest())
            self.assertFalse(published['wandb_enabled'])
            write_summaries(root/'results')
            self.assertTrue((root/'results/exploratory_test_summary.csv').exists())
            self.assertFalse((root/'results/paper_summary.csv').exists())


if __name__ == "__main__":
    unittest.main()
