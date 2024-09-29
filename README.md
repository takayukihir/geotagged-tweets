# geotagged_tweets
This repository contains code used in the research presented in the following paper: Takayuki Hiraoka, Takashi Kirimura, Naoya Fujiwara (2024) "Geospatial analysis of toponyms in geo-tagged social media posts".

```
/project_root
│
├── original/           # Original data
├── code/               # Source code and Jupyter notebooks for data analysis
├── figure/             # Figure folder
├── environment.yml     # Environment YAML file 
└── README.md           # Project README file
```

The necessary depedencies are described in `environment.yml`. Please install them with your favorite package manager.
For example:

```bash
micromamba create -f environment.yml
```

Each Jupyter notebook in `code` directory can be used for reproducing the results presented in the paper. To run them, download the dataset from Zenodo and create `original` directory. To run `code/make_figs4-7.ipynb`, unzip `code/ipaexg00401.zip`. 