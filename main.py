import os
import nemo
import nemo.collections.asr as nemo
import pytorch_lightning as pl
from ruamel.yaml import YAML
from omegaconf import DictConfig

trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=50)



config_path = './configs/config.yaml'
yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)

params['model']['train_ds']['manifest_filepath'] = "./dataset/train_manifest.json"
params['model']['validation_ds']['manifest_filepath'] = "./dataset/test_manifest.json"

model = nemo.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)


trainer.fit(model)


model.setup_test_data(test_data_config=params['model']['validation_ds'])
model.cuda()
model.eval()


wer_nums = []
wer_denoms = []

for test_batch in model.test_dataloader():
    test_batch = [x.cuda() for x in test_batch]
    targets = test_batch[2]
    targets_lengths = test_batch[3]
    log_probs, encoded_len, greedy_predictions = model(
        input_signal=test_batch[0], input_signal_length=test_batch[1]
    )
    # Notice the model has a helper object to compute WER
    model._wer.update(greedy_predictions, targets, targets_lengths)
    _, wer_num, wer_denom = model._wer.compute()
    model._wer.reset()
    wer_nums.append(wer_num.detach().cpu().numpy())
    wer_denoms.append(wer_denom.detach().cpu().numpy())

    # Release tensors from GPU memory
    del test_batch, log_probs, targets, targets_lengths, encoded_len, greedy_predictions

print(f"WER = {sum(wer_nums) / sum(wer_denoms)}")
