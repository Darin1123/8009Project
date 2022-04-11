from load import DataLoader
from diffusion_feature import preprocess
from logger import Logger
from models import MLPLinear, MLP
import sys
import gc
from outcome_correlation import *
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import glob


###########################
#      choose model       #
###########################
# model related
MODEL = 'linear'  # 'lp' | 'plain' | 'linear' | 'mlp'
USE_EMBEDDINGS = False  # True | False
NUM_LAYERS = 3  # number of layers in the MLP
HIDDEN_CHANNELS = 256

############################
#      test settings       #
############################
# test related
RUNS = 2  # number of runs, should be >=2
LR = 0.01  # learning rate
EPOCHS = 300
# constants
DEVICE = 'cpu'  # computation device
DATASET = 'arxiv'

##########################################
#      DO NOT MODIFY THE REMAINING       #
##########################################
# load data
dataLoader = DataLoader()
dataset, split_idx = dataLoader.load_data()
preprocess_data = dataLoader.load_preprocess_data()
print("[INFO] successfully load data!")

# components
evaluator = Evaluator(name=f'ogbn-{DATASET}')
logger = Logger(RUNS)


def train(model, x, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x[train_idx])
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def run_test(model, x, y, split_idx, evaluator):
    model.eval()

    out = model(x)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return (train_acc, valid_acc, test_acc), out


# device
device = torch.device(DEVICE)

# data
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
x = data.x


def eval_test(result, idx=split_idx['test']):
    return evaluator.eval({'y_true': data.y[idx], 'y_pred': result[idx].argmax(dim=-1, keepdim=True), })['acc']


# lp
if MODEL == 'lp':
    dataset = PygNodePropPredDataset(name=f'ogbn-{DATASET}', root='data')
    data = dataset[0]
    adj, D_isqrt = process_adj(data)
    normalized_adjs = gen_normalized_adjs(adj, D_isqrt)
    AD = normalized_adjs[2]
    lp_dict = {
        'idxs': ['train'],
        'alpha': 0.9,
        'num_propagations': 50,
        'A': AD,
    }
    out = label_propagation(data, split_idx, **lp_dict)
    print('Valid acc: ', eval_test(out, split_idx['valid']))
    print('Test acc:', eval_test(out, split_idx['test']))
    sys.exit()

# embedding
if USE_EMBEDDINGS:
    embeddings = torch.cat([preprocess(preprocess_data, 'diffusion', post_fix=DATASET)], dim=-1)
    x = torch.cat([x, embeddings], dim=-1)

# model
if MODEL == 'mlp':
    model = MLP(
        x.size(-1), HIDDEN_CHANNELS, dataset.num_classes, NUM_LAYERS,
        0.5, DATASET == 'products'
    ).cpu()
elif MODEL == 'linear':
    model = MLPLinear(x.size(-1), dataset.num_classes).cpu()
elif MODEL == 'plain':
    model = MLPLinear(x.size(-1), dataset.num_classes).cpu()
else:
    print("the model is not defined in this implementation")
    sys.exit()

x = (x - x.mean(0)) / x.std(0)
x = x.to(device)
y_true = data.y.to(device)
train_idx = split_idx['train'].to(device)

model_dir = prepare_folder(f'{DATASET}_{MODEL}', model)

# generate models
for run in range(RUNS):
    gc.collect()
    print(f'# parameters: {sum(p.numel() for p in model.parameters())}')
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_valid = 0
    best_out = None
    for epoch in range(1, EPOCHS):
        loss = train(model, x, y_true, train_idx, optimizer)
        result, out = run_test(model, x, y_true, split_idx, evaluator)
        train_acc, valid_acc, test_acc = result
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_out = out.cpu().exp()

        print(f'Run: {run + 1:02d}, '
              f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * test_acc:.2f}%')
        logger.add_result(run, result)

    logger.print_statistics(run)
    torch.save(best_out, f'{model_dir}/{run}.pt')

# result
logger.print_statistics()

# experiments
dataset = PygNodePropPredDataset(name=f'ogbn-{DATASET}', root='data')
data = dataset[0]
adj, D_isqrt = process_adj(data)
normalized_adjs = gen_normalized_adjs(adj, D_isqrt)
DAD, DA, AD = normalized_adjs
model_outs = glob.glob(f'models/{DATASET}_{MODEL}/*.pt')
lp_dict = {
    'idxs': ['train'],
    'alpha': 0.9,
    'num_propagations': 50,
    'A': AD,
}
plain_dict = {
    'train_only': True,
    'alpha1': 0.87,
    'A1': AD,
    'num_propagations1': 50,
    'alpha2': 0.81,
    'A2': DAD,
    'num_propagations2': 50,
    'display': False,
}
plain_fn = double_correlation_autoscale

"""
If you tune hyper-parameters on test set
{'alpha1': 0.9988673963255859, 'alpha2': 0.7942279952481052, 'A1': 'DA', 'A2': 'AD'} 
gets you to 72.64
"""
linear_dict = {
    'train_only': True,
    'alpha1': 0.98,
    'alpha2': 0.65,
    'A1': AD,
    'A2': DAD,
    'num_propagations1': 50,
    'num_propagations2': 50,
    'display': False,
}
linear_fn = double_correlation_autoscale

"""
If you tune hyperparameters on test set
{'alpha1': 0.9956668128133523, 'alpha2': 0.8542393515434346, 'A1': 'DA', 'A2': 'AD'}
gets you to 73.35
"""
mlp_dict = {
    'train_only': True,
    'alpha1': 0.9791632871592579,
    'alpha2': 0.7564990804200602,
    'A1': DA,
    'A2': AD,
    'num_propagations1': 50,
    'num_propagations2': 50,
    'display': False,
}
mlp_fn = double_correlation_autoscale

gat_dict = {
    'labels': ['train'],
    'alpha': 0.8,
    'A': DAD,
    'num_propagations': 50,
    'display': False,
}
gat_fn = only_outcome_correlation

if MODEL == 'plain':
    evaluate_params(data, eval_test, model_outs, split_idx, plain_dict, fn=plain_fn)
elif MODEL == 'linear':
    evaluate_params(data, eval_test, model_outs, split_idx, linear_dict, fn=linear_fn)
elif MODEL == 'mlp':
    evaluate_params(data, eval_test, model_outs, split_idx, mlp_dict, fn=mlp_fn)
