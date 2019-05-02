# CSCE670
#### Lin 2019-04-17:  

###### Include GRU4REC in repository.   
I used GRU4REC framework to predict next item using current item in hope of
finding some time relation between items.   
I modified the way GRU4REC does negative sampling. Negative sample are selected
randomly from training set. Unlike GRU4REC sampled them from the same batch. 

###### How to run:
    1. pip install -r requirements.txt
    2. Use preprocess.py to separate train.csv to train_tr.csv and validation.csv
    3. run main.py with desired hyper-param.

###### Further Improvement
    1. Embed action_type and concate with reference embedding.
    2. Sample more negative sample. As pointed out in the paper, more negative sample can improve performance a lot.
       But using all samples will need tremendous amount of memory.
    3. Increase hidden unit number.
    4. Try TOP1 max or BPR max loss.

#### Henry:

script/validate.py: use for offline checking the performance by sampling some sessions and evaluate on training data.

script/other_scrtipts: for feature extraction

script/rerank.py: reranking via xgboost

Note: For now, I only extract the user-item relations and construct the user-item graph(mixed.edgelist) by some heuristics, and apply some GNN algorithm for learning the item representations.

For recommendation scenario, I apply the concept of item-based KNN. 
Based on the learned item representation from GNN algorithm, 
I make recommendations for each session  by calculating the similarity
in the learned embedding space for the items in the session, and choose the top-k
most similar items for the recommendation list.

# Performance of Embedding Algorithm
HopRec: 0.4859
MF: 0.4752
APP: 0.4657
HPE: 0.4498
Deepwalk: 0.4151
Item-Features only: 0.4398


P.S: We need to find a way to share the features, they are too big to share on github.

#### Chao:

Metadata Preprocessing.ipynb: extracting the item features from metadata, and convert the features into a metrix for all items. It also extracts the locations and user interactions as the features for items. 

2layer_30d_AutoEncoder.py: a two-layer autoencoder that can reduce the dimentionality of the item feature matrix into a 30-dimensional matrix.

#### Qing:
I used autoencoder method to extract 10 dimensional embedding features from item metadata, the loss is 0.03 which is acceptable.

The feature is stored in .csv file.
