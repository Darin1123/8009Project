# 8009 Project

> April 7, 2022

## Some Links

### Project

- [Project Submission](https://canvas.cityu.edu.hk/courses/46749/assignments/196816)
- [README Sample](https://github.com/rguo12/network-deconfounder-wsdm20)

### Dataset

- [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)

### Code Related

- [Load Data Code](https://github.com/rguo12/network-deconfounder-wsdm20)

#### Issues

##### 1. ImportError: cannot import name 'PygGraphPropPredDataset' from 'ogb.graphproppred']

[Solution](https://github.com/snap-stanford/ogb/issues/115) -> [Solution](https://github.com/pyg-team/pytorch_geometric#installation)


## Objective

Predict the class for each paper from the dataset.

## Dataset Info

Here is the link to the [Dataset](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) and here is the link to [leaderboard](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv).

**Graph**: The ogbn-arxiv dataset is a directed graph, representing the citation network between all Computer Science (CS) arXiv papers indexed by MAG [1]. Each node is an arXiv paper and each directed edge indicates that one paper cites another one. Each paper comes with a 128-dimensional feature vector obtained by averaging the embeddings of words in its title and abstract. The embeddings of individual words are computed by running the skip-gram model [2] over the MAG corpus. We also provide the mapping from MAG paper IDs into the raw texts of titles and abstracts [here](https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz). In addition, all papers are also associated with the year that the corresponding paper was published.

**Prediction task**: The task is to predict the 40 subject areas of arXiv CS papers, e.g., cs.AI, cs.LG, and cs.OS, which are manually determined (i.e., labeled) by the paper’s authors and arXiv moderators. With the volume of scientific publications doubling every 12 years over the past century, it is practically important to automatically classify each publication’s areas and topics. Formally, the task is to predict the primary categories of the arXiv papers, which is formulated as a 40-class classification problem.

**Dataset splitting**: We consider a realistic data split based on the publication dates of the papers. The general setting is that the ML models are trained on existing papers and then used to predict the subject areas of newly-published papers, which supports the direct application of them into real-world scenarios, such as helping the arXiv moderators. Specifically, we propose to train on papers published until 2017, validate on those published in 2018, and test on those published since 2019.


## Approach

The core method we use is C&S by following the paper ["Combining Label Propagation and Simple Models Out-performs Graph Neural Networks"](https://arxiv.org/abs/2010.13993). We tried three different models and embedding to find a good model.


## Dependencies

```text
Python 3.9.12
Pytorch 1.11.0
Numpy 1.22.3
Optuna 2.10.0
```

## Datasets

The data could be found in the data folder. If it is not there, the data could be downloaded by running the code.

## Running the experiment

In folder src, simply run:

```bash
python3 main.py
```

To select a different algorithm, please modify the parameters listed on the top of `main.py`.

> **Possible issue**
> 
> By running the above command, python may get stuck and produce nothing. To solve this problem, you could use [PyCharm](https://www.jetbrains.com/pycharm/) to run the code.

## References

- [Correct and Smooth](https://github.com/CUAI/CorrectAndSmooth)
- [Combining Label Propagation and Simple Models Out-performs Graph Neural Networks](https://arxiv.org/abs/2010.13993)
