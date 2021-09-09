# Project :

This project is about the commercial segmentation of an e-commerce site named "Olist", the goal was to create a segmentation that is easily interpretable by the marketing team.


# Requierement :

- conda

# Data :

Data use on this project is on <a href="https://www.kaggle.com/olistbr/brazilian-ecommerce/download">download here</a>.<br>

unzip file and put on "/Solution/data/"


# Conda environment :

## How to use :

Create environment : <br>
`conda env create --file environment.yaml` <br>

Activate environment : <br>
`conda activate projet5`<br>

Exit current environment : <br>
`conda deactivate projet5`

Install requirement file :<br>
`pip install -r requirement.txt`


## Add environment to Jyputer notebook :

This tells jupyter to take the current environment("projet5")<br>
`python -m ipykernel install --user --name=projet5`

# Files content :

This project was structured in 2 parts, a file that we will call "ETL" which is the preparation of the final file from the different initial data source (`POLIST_01_notebookanalyse.ipynb`).<br>
And the second main file is about the marketing segmentation and further analysis of the final dataset (`POLIST_02_notebookanalysis.ipynb`).<br>
<br>
There are also 2 other files containing functions used in the project.
