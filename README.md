# CSCE670
script/validate.py: use for offline checking the performance by sampling some sessions and evaluate on training data.

script/other_scrtipts: for feature extraction

Henry:

Note: For now, I only extract the user-item relations and construct the user-item graph(mixed.edgelist) by some heuristics, and apply some GNN algorithm for learning the item representations.

For recommendation scenario, I apply the concept of item-based CF. 
Based on the learned item representation from GNN algorithm, 
I make recommendations for each session  by calculating the similarity
in the learned embedding space for the items in the session, and choose the top-k
most similar items for the recommendation list.


P.S: We need to find a way to share the features, they are too big to share on github.
