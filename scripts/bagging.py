import os
from haven import haven_utils as hu
import sys
sys.path.append('/home/xhdeng/projects/def-dnowrouz-ab/xhdeng/Cassava-Classification/src')
import models
sys.path.append('/home/xhdeng/projects/def-dnowrouz-ab/xhdeng/Cassava-Classification/datasets')
import datasets
import torch
import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def val_on_loader(loader, model_list):

        acc_sum = 0.0
        acc_samples = 0.
        results = 'label, pred, fname\n'
        for (images, labels, fnames) in tqdm.tqdm(loader, desc='Validating'): 
            image = images.to(DEVICE)
            
            votes = torch.zeros([20, 5])
            for model in model_list:
                model.network.eval()
                logits = model.network(image)     
                preds = logits.argmax(dim=1)
                preds_cpu = preds.cpu()
                votes[range(len(preds_cpu)), preds_cpu] += 1

            for i in range(20):
                print('%s, %s, %s\n' % (labels[i], votes.argmax(dim=1)[i], fnames[i]))
                result = '%s, %s, %s\n' % (labels[i], votes.argmax(dim=1)[i], fnames[i])
                results += result
        # hu.save_txt('/home/xhdeng/scratch/results/debug/cassava_debug/false_positive.csv', s)
        hu.save_txt('/home/xhdeng/scratch/results/debug/cassava_debug/bagging_val_result.csv', results)

# load model
exp_ids = ['4b46b0cf6b73b2df96e57ebb2ee104fa', 'abe0c201043742b6ab4e1de0032a0255', 'ea04aa8bdf4d9306464f8db5d06505d4']
model_list = []
exp_dict = {}
for exp_id in exp_ids:
    base_dir = os.path.join('/home/xhdeng/scratch/results/debug/cassava/', exp_id)
    exp_dir = os.path.join(base_dir, 'exp_dict.json')
    exp_dict = hu.load_json(exp_dir)
    exp_dict['val_transform'] = {'name': 'default', 'im_size': 256}
    model = models.get_model(exp_dict)
    model_dir = os.path.join(base_dir, 'model.pth')
    model.network.load_state_dict(hu.torch_load(model_dir, map_location=torch.device(DEVICE)))
    model_list.append(model)

data_dir = '/home/xhdeng/scratch/datasets/cassava'
exp_dict['batch_size'] =  20
val_loader = datasets.get_loader(split="val",
                                     exp_dict=exp_dict,
                                     datadir=data_dir)
val_on_loader(val_loader, model_list)




