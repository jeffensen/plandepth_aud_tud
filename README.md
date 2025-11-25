# Forward planning strategies in Alcohol and Tobacco Use Disorder
Investigating differences in forward planning during sequential decision-making between healthy controls and participants with alcohol and/or tobacco use disorder using the [Space Adventure Task](https://github.com/expfactory-experiments/space_adventure_pd) adapted from [https://github.com/dimarkov/sat](https://github.com/dimarkov/sat) and
a value iteration model described in https://github.com/dimarkov/pybefit/tree/master/examples/plandepth. The model originally entailed planning depth limitation and was extended with additional planning strategies like low-probability pruning and probability discounting.

Requirements
------------
    pyro
    pytorch
    numpy
    pandas
    matplotlib
    seaborn
    
Usage
------------
Running the following scripts performs parameter fitting, model comparison, model based data analysis and generation of relevant figures:

| Nr. | Script  | Description |
| ------------- | ------------- | ------------- |
| 1  |  data_preprocessing.ipynb  | ...preprocesses raw data from redcap questionaires and raw data of all tasks. performs exlusion and propensity score matching of healthy control.  |
| 2  |  fitting_model_to_behaviour_*.ipynb  | ...performs the inference of model parameters of respective model from Space Adventure Task raw data, merges with preprocessed data and outputs tidy overall .CSV dataset.|
| 3  |  fitting_model_comparison.ipynb  | ...performs model comparison of the planning strategy models.|
| 4  |  data_analysis_LPP_and_probDiscounting.ipynb  | ...imports .CSV dataset and generates results plots.  |
| 4  |  data_analysis.sps  | ...imports .CSV dataset and performs all statistical analyses in SPSS.  |
