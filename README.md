# Performing Gender Through Dialogue

###

This repository contains code for embeddings, analysis, plots and results of our article: 

Lucas van der Deijl & Alie Lassche, "Performing Gender Through Dialogue. A Computational Approach to Male and Female Speech in Dutch Drama (1600-1800)" in _Early Modern Low Countries_ (forthcoming).

## Useful directions 📌

Some useful directions:
- `emlc-performing-gender/` the main folder contains the Python files to get the results
- `/notebooks/` contains the notebooks used for the analysis
- `/results/` contains results of the experiments


## Data & article 📝

Please cite our [article](link-coming-soon) if you use the code or the embeddings.



## Project Organization 🏗️

```
├── LICENSE                    <- Open-source license if one is chosen.
│
├── README.md                  <- The top-level README for developers using this project.
│
├── data_preparation.py        <- Python file for parsing XML files and getting speech text and gender per character.
│
├── clustering_task.py         <- Python file for clustering task to choose the most fitting Transfomer model.
│
├── get_embeddings.py          <- Python file for embedding inference.
│
├──  data/                     <- Contains the file created with data_preparation.py.
│
├── notebooks/                 <- Jupyter notebooks.
│      │
│      └── train_classifiers.ipynb          			      <- Jupyter notebook for training classifiers with TF-IDF and embeddings as features.
│      │
│      └── character_analysis_drama_corpus_TEI.ipynb    <- Jupyter notebook for extracting data from a TEI-encoded corpus of plays and compute descriptive statistics.
│
├── figs/                      <- Figures from the article. 
│
└──  results/                  <- Results from machine learning experiments.
```
