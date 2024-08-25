# 7/30/24

- Determine why the number of non-cyberbullying entries in Dataset 2 has changed
- Perform experiment using primarily the DQE data in training. Testing can be done with both other datasets!!!
- Perform CV properly for the inital testing on Dataset 1. Run these multiple times as embedding methods such as BOW can cause poor results to due important words not being "in the bag". 

### Experiment 1

1. Split Dataset 1 into train and test.
2. Perform RandomCV search on the train portion.
3. Train a model and test it. 

### Experiment 2

1. Write a script that gets data primarily obtained via DQE.
    2. Add data from Dataset 2 to the Non-Cyberbullying class.
2. Use CV testing to determine the best hyperparameters for this data. 
3. Train a model and test it on Dataset 2 and 3 (ensure no data leakage!)

Compare the results of the two experiments!

### Stats

Assuming Uniform Sampling
               Not CB    Age,  Ethnicity,  Gender,  Religion, Other CB
From Dataset:  8000      132,    171,      5,051    1651,     7637
From DQE:      0         7868,   7829,     2949     6349      363

Not from DQE 22642
From DQE     25358


### Cyberbullying / Hate Speech Definitions

**TweetBLM**: 
Hateful text: This category of tweets
containing information that is broadly defined as
a form of expression that “attacks or diminishes,
that incites violence or hate against groups, based
on specific characteristics such as physical appearance, religion, descent, national or ethnic origin,
sexual orientation, gender identity, or other,and it
can occur with different linguistic styles, even in
subtle forms or when humor is used (Fortuna and
Nunes, 2018)”.


**8/5/24**
- Verfiy that experiments were performed properly 
- Find a test dataset
- Start RandomCV searches for Dataset 1 

Experimentation Ideas
- Keep punctuation
- Use bigrams, trigrams, etc


