import os
from haven import haven_utils as hu
import sys
sys.path.append('/home/xhdeng/projects/def-dnowrouz-ab/xhdeng/Cassava-Classification/src')
import models
sys.path.append('/home/xhdeng/projects/def-dnowrouz-ab/xhdeng/Cassava-Classification/datasets')
import datasets
import torch
import tqdm

exp_id = 'ea04aa8bdf4d9306464f8db5d06505d4'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def val_on_loader(loader, network, exp_id, exp_dict):
        network.eval()

        acc_sum = 0.0
        acc_samples = 0.
        results = 'label, pred, fname\n'
        for (images, labels, fnames) in tqdm.tqdm(loader, desc='Validating'): 
            image = images.to(DEVICE)
            
            logits = network(image)     
            preds = logits.argmax(dim=1)
            preds_cpu = preds.cpu()
            for i in range(exp_dict['batch_size']):
                result = '%s, %s, %s\n' % (labels[i], preds_cpu[i], fnames[i])
                results += result
        # hu.save_txt('/home/xhdeng/scratch/results/debug/cassava_debug/false_positive.csv', s)
        hu.save_txt('/home/xhdeng/scratch/results/debug/cassava_debug/%sval_result.csv' % exp_id, results)


# laod model
base_dir = os.path.join('/home/xhdeng/scratch/results/debug/cassava/', exp_id)
exp_dir = os.path.join(base_dir, 'exp_dict.json')
exp_dict = hu.load_json(exp_dir)
exp_dict['val_transform'] = {'name': 'default', 'im_size': 256}

model = models.get_model(exp_dict)
model_dir = os.path.join(base_dir, 'model.pth')
model.network.load_state_dict(hu.torch_load(model_dir, map_location=torch.device(DEVICE)))

data_dir = '/home/xhdeng/scratch/datasets/cassava'
val_loader = datasets.get_loader(split="val",
                                     exp_dict=exp_dict,
                                     datadir=data_dir)

val_on_loader(val_loader, model.network, exp_id, exp_dict)