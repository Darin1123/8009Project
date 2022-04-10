from load import DataLoader
from diffusion_feature import preprocess
from logger import Logger
from models import MLPLinear, MLP
import sys
import gc
from outcome_correlation import *
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


###########################
#      choose model       #
###########################
# model related
MODEL = 'plain'  # 'lp' | 'plain' | 'linear' | 'mlp'
USE_EMBEDDINGS = True  # True | False
NUM_LAYERS = 3  # number of layers in the MLP
HIDDEN_CHANNELS = 256


############################
#      test settings       #
############################
# test related
RUNS = 5  # number of runs, should be >=2
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


def prepare_folder(name, model):
    model_dir = f'models/{name}'

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    with open(f'{model_dir}/metadata', 'w') as f:
        f.write(f'# of params: {sum(p.numel() for p in model.parameters())}\n')
    return model_dir


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

# embedding
if USE_EMBEDDINGS:
    embeddings = torch.cat([preprocess(preprocess_data, 'diffusion', post_fix=DATASET)], dim=-1)
    x = torch.cat([x, embeddings], dim=-1)

# model
model = None


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

# compute
for run in range(RUNS):
    gc.collect()
    print(sum(p.numel() for p in model.parameters()))
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
