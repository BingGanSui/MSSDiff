import torch

def extract(exp_name, dataset, epoch):
    s = torch.load(f'results/{exp_name}/checkpoints/checkpoint-{epoch}.pt', map_location=torch.device('cpu'))
    t = {}
    for k, v in s['model'].items():
        if k.startswith('model.'):
            t[k[6:]] = v
    torch.save(t, f'pretrained-rrdbnet-{dataset}.pt')


pairs = [
    ['<EXPERIMENT NAME HERE>', '<DATASET NAME HERE>','<EPOCH HERE>'],
]
for pair in pairs:
    extract(*pair)
