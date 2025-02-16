# geotagged_tweets
This repository contains code used in the research presented in the following paper: Takayuki Hiraoka, Takashi Kirimura, Naoya Fujiwara (2024) "Geospatial analysis of toponyms in geo-tagged social media posts" 	[arXiv:2410.03250](https://doi.org/10.48550/arXiv.2410.03250).

```
/project_root
│
├── original/    # Original data
│   ├── japan_MetropolitanEmploymentArea2015map/    # Shapefiles of metropolitan areas
│   ├── ward_shapefiles/    # Shapefiles of wards
├── scratch/    # Derived data
│   ├── preprocessed_mcntlt7_selected    # Tweet count data
│   ├── population/    # Population data
├── code/    # Source code and Jupyter notebooks for data analysis
├── figure/    # Figure folder
├── environment.yml    # Environment YAML file 
└── README.md    # Project README file
```

The necessary depedencies are described in `environment.yml`. Please install them with your favorite package manager.
For example:

```bash
micromamba create -f environment.yml
```

Each Jupyter notebook in `code` directory can be used for reproducing the results presented in the paper. To run them, download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.13860968) and create `original` and `scratch` directory. To run `code/make_figs4-7.ipynb`, unzip `code/ipaexg00401.zip`. 