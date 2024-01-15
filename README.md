# MADS-exam-24

This year, the junior has learned quit a lot about machine learning.
He is pretty confident he wont make the same mistakes as last year; this year, he has helped you out by doing some data exploration for you, and he even created two models!

However, he didnt learn to hypertune things, and since you are hired as a junior+ datascientist and he has heard you had pretty high grades in the machine learning course, he has asked you to help him out.

## The data
We have two datasets:
### The PTB Diagnostic ECG Database

- Number of Samples: 14552
- Number of Categories: 2
- Sampling Frequency: 125Hz
- Data Source: Physionet's PTB Diagnostic Database

All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 187. There is a target column named "target".

You can find this in `data/heart_train.parq` and `data/heart_test.parq`.

### Arrhythmia Dataset

- Number of Samples: 109446
- Number of Categories: 5
- Sampling Frequency: 125Hz
- Data Source: Physionet's MIT-BIH Arrhythmia Dataset
- Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 187. There is a target column named "target".

You can find this in `data/heart_big_train.parq` and `data/heart_big_test.parq`.

## Exploration
In `notebooks/01_explore-heart.ipynb` you can find some exploration. It's not pretty, and
he hasnt had time to add a lot of comments, but he thinks it's pretty self explanatory for
someone with your skill level.

## Models
There are two notebooks with models. He has tried two approaches: a 2D approach, and a 1D approach. Yes, he is aware that this is a time series, but he has read that 2D approaches can work well for timeseries, and he has tried it out.

## Your task

You can find the models in `src/models.py`, but they are also in the notebooks themselves. Your task is to:
1. Explore the models manually, to get some intuition about their performance
2. Based on your knowlodge about hypertuning, you might want to change and expand the models: add more parameters you might want to tune, add additional layers, and if you feel like it you could also create additional models that you think might be promising.
3. When you are happy with your manual research, set up a hypertune.py file and hypertune the models with Ray. Make sure you log everything (eg with ray, mlflow of gin-config).
4. Create a notebook where you explore your hypertuning results, and save the plots that are most relevant for your research.
5. Create a short summary (2 pages) of your results, where you show the relevant hyperparameters you have tuned. Yes, I know you did more than the summary, but I want to see that you can summarize your results in a clear way. You can take inspiration from table 3 in the (attention is all you need)[https://arxiv.org/pdf/1706.03762.pdf] paper for an excelent summary of hypertuning. Export your summary as pdf in your repo.

Because this is a medical dataset, an we are trying to spot disease, the client thinks it is more important that you find as much sick people as possible, even if that means you will have more false positives. So, they want you to optimize for recall, more than for precision.

You will be graded for:

- Overall presentation and clarity of your work. (10%)
- The level of your change and expansion of the models: balance performance (exploit) and creativity/curiosity (explore). (30%)
- The relevance of your starting hyperparameters, based on your manual hypertuning. (10%)
- The quality of your implementation of the hypertuning in hypertune.py (20%)
- The clarity and relevance of your summary (30%)

During the oral exam we will discuss your work, and you will have to explain your choices.
