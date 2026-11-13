"""Disposable untrained-model evaluation-path test; never publishes model metrics."""
import json
import gc
import sys
import torch
from panda_unified_campaign import plan, assets, model, training, evaluation, selection
from panda_unified_campaign.release import verify
from panda_stage2_condq.training import configure_precision
from ta2.evaluation import evaluator_identity

configure_precision()
release = verify(environment=True)
proof = {'kind': 'UNTRAINED_DISPOSABLE_EVALUATION_PATH_SMOKE',
         'source_release': release['source_sha256'], 'published': False, 'checks': []}
for server in sys.argv[1:]:
    source, registered = assets.sources(server)
    for case in plan.stage1_cases(server):
        cfg = plan.resolve(server, case, 1)
        net = model.Stage1(cfg, registered.inventory.manifest()).cuda()
        trainer = training.Trainer(cfg, net)
        # Only a protocol-boundary fixture: no trained result, weight or evaluation
        # record is saved, and there is no call to production publication.
        trainer.updates = 1000
        record = evaluation.evaluate(trainer, source, evaluator_identity())
        evaluation.verify_record(record, cfg)
        inventory = selection.validation_inventory(source, cfg['reference_mode'])
        val = evaluation.validation(trainer, source, inventory)
        proof['checks'].append({'server_recipe': server, 'case': case,
            'RR_scenes': len(record['rr']['per_scene']), 'FR_scenes': len(record['fr']['per_scene']),
            'validation_samples': val['sample_count'], 'finite_Val_ERGAS': bool(torch.isfinite(torch.tensor(val['Val_ERGAS']))),
            'checkpoint_saved': False})
        print(json.dumps(proof['checks'][-1]), flush=True)
        del trainer, net
        gc.collect(); torch.cuda.empty_cache()
print(json.dumps(proof), flush=True)
