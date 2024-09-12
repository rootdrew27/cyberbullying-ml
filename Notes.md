### Dataset Stats

Assuming Uniform Sampling
               Not CB    Age,  Ethnicity,  Gender,  Religion, Other CB
From Dataset:  8000      132,    171,      5,051    1651,     7637
From DQE:      0         7868,   7829,     2949     6349      363

Not from DQE 22642
From DQE     25358


### Cyberbullying / Hate Speech Definitions

**8/5/24**
- Verfiy that experiments were performed properly 
- Find a test dataset
- Start RandomCV searches for Dataset 1 

Experimentation Ideas
- Keep punctuation
- Use bigrams, trigrams, etc

**8/26/24**
- Modify the CatBoostClassifier tests so that they save all catboost parameters (i.e. `m.get_all_params()`)

**9/3/24**
- Modify the XGBClassifier tests so that they save all parameters (i.e. `m.get_all_params()`)
- Create list of papers to use for literature review
- Get best Hyps for exp4
- Run Exp3
- Run Exp4

**9/4/24**
- Find 5 sets of hyperparameters for each model (via 5 fold CV) Use GridSearchCV
- Test each set of hyperparameters on the test set
- Compare the performance of each model

- Create list of papers to use for literature review
    - https://arxiv.org/pdf/2402.16458 (mentions "swear word bias" )

- Randomly remove swear words from the dataset during training (this should help de-bias the dataset)

- Types of Bias in NLP ([Paper](https://compass.onlinelibrary.wiley.com/doi/10.1111/lnc3.12432))
    - Data Selection Bias
    - Label Bias
    - etc
Additionally, bias can be introduced via unsupervised labeling, and through feature selection.
Data Selection Bias is extremely relevant to the field of cyberbullyind detection, as the collection of many datasests is facilitated by querying techniques which favor particular keywords, for example, by querying for words commonly associated with hate. It would be understandable for a model to associate hateful words with a higher degree of hate speech, but hate speech which does not involve these words would be underrepresented. This may seem paradoxical, but a tweet such as "All brown men should politely leave this country before they deal with the consequences", might not be classified as hate speech, as it does not contain any hateful words. However, this does not mean that the statement is not hateful, or not harmful. 

Generalizability
Cross-Dataset: Cross-Domain & Cross-Platform



Semi-Supervised Learning
- Through the use of an query algorithm and labeled data, a larger dataset is composed. This larger dataset is believed to be more adaquate for training a model.

- This method makes the assumption that their algorithm will accuractely label data the majority of the time.

- Many of the methods used in semi-supervised learning rely on the consistency assumption, the low-density separation assumption, and the manifold assumption.

Semi-Supervised Learning methods are domain specific! 







