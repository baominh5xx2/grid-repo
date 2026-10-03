import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn
from gated_dual_ema_msd.cli import evaluate


class ArchitectureFinalEvaluationTests(unittest.TestCase):
    def test_setup_failure_can_retry_but_started_inference_cannot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp); checkpoint=root/'best.pt'; model=nn.Linear(1,3)
            torch.save(model.state_dict(),checkpoint)
            args=['--checkpoint',str(checkpoint),'--dataset','vinli','--frozen_final',
                  '--output_csv',str(root/'final.csv')]
            with patch.object(evaluate,'load_nli_dataset',side_effect=RuntimeError('download failed')):
                with self.assertRaisesRegex(RuntimeError,'download failed'): evaluate.main(args)
            ledger=json.loads((root/'final.json').read_text())
            self.assertEqual((ledger['state'],ledger['test_evaluations']),('initializing',0))
            with patch.object(evaluate,'load_nli_dataset',return_value=({'test':[]},object())), patch.object(evaluate,'create_nli_model',return_value=model), patch.object(evaluate,'evaluate_model',side_effect=RuntimeError('inference interrupted')) as inference:
                with self.assertRaisesRegex(RuntimeError,'inference interrupted'):
                    evaluate.main(args+['--resume_unstarted'])
                inference.assert_called_once()
                with self.assertRaisesRegex(ValueError,'already|started'):
                    evaluate.main(args+['--resume_unstarted'])
                inference.assert_called_once()

    def test_test_stays_locked_before_loading_data_or_model(self):
        with patch.object(evaluate, 'load_nli_dataset') as loader:
            with self.assertRaisesRegex(ValueError,'locked'):
                evaluate.main(['--checkpoint','unused.pt','--dataset','vinli'])
            loader.assert_not_called()

    def test_final_inference_includes_test_and_writes_once_only_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp); checkpoint=root/'best.pt'
            model=nn.Linear(1,3);torch.save(model.state_dict(),checkpoint)
            frame=pd.DataFrame([dict(sample_id='x',gold_label='E',pred_label='E')])
            def load(dataset, **kwargs):
                return ({'test':[{'id':'x'}]} if kwargs.get('include_test') else {'dev':[{'id':'d'}]}), object()
            args=['--checkpoint',str(checkpoint),'--dataset','vinli','--frozen_final',
                  '--output_csv',str(root/'final.csv')]
            with patch.object(evaluate,'load_nli_dataset',side_effect=load) as loader, patch.object(evaluate,'create_nli_model',return_value=model), patch.object(evaluate,'evaluate_model',return_value=({'macro_f1':1.,'accuracy':1.},frame)) as inference:
                evaluate.main(args)
                self.assertTrue(loader.call_args.kwargs['include_test'])
                inference.assert_called_once()
                self.assertEqual(inference.call_args.kwargs['max_length'],512)
                ledger=json.loads((root/'final.json').read_text())
                self.assertEqual(ledger['test_evaluations'],1)
                self.assertEqual(ledger['eval_precision'],'fp32')
                self.assertEqual(len(ledger['checkpoint_sha256']),64)
                with self.assertRaisesRegex(ValueError,'already|exists'):
                    evaluate.main(args)
                self.assertEqual(inference.call_count,1)

    def test_dev_evaluation_never_loads_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=pathlib.Path(tmp)/'best.pt'; model=nn.Linear(1,3)
            torch.save(model.state_dict(),checkpoint)
            with patch.object(evaluate,'load_nli_dataset',return_value=({'dev':[{'id':'d'}]},object())) as loader, patch.object(evaluate,'create_nli_model',return_value=model), patch.object(evaluate,'evaluate_model',return_value=({'macro_f1':1.},pd.DataFrame())):
                evaluate.main(['--checkpoint',str(checkpoint),'--dataset','vinli','--split','dev'])
                self.assertFalse(loader.call_args.kwargs['include_test'])


if __name__ == '__main__': unittest.main()
