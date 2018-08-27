# FewShotAuthorClassificationMetaLearning
With Reptile https://blog.openai.com/reptile/ I attempt to meta learn few shot author classification.
Currently tested the viability on the Gutenberg dataset https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html
as well as on this specific reddit dataset https://github.com/linanqiu/reddit-dataset.

The models wers trained for 2 days on the Guttenberg dataset on a Google Compute instance with a V100 attached and for 3 hours on the reddit dataset.

# Results
Some encouraging results were obtained on both the reddit comment dataset and the Guttenberg dataset.
## Reddit dataset
Using as inputs 320 tokens worth of comments from an out of sample redditor in the reddit dataset, a 50 way top 1 accuracy of 92% was obtained on unseen data.
![alt text](https://github.com/OctThe16th/FewShotAuthorClassificationMetaLearning/raw/master/images/AccuracyReddit50Way.PNG)
## Guttenberg dataset
Using as inputs 5120 tokens worth of text taken from different books from an out of sample author, a 5 way top 1 accuracy of 55% was obtrained on unseen data.
![alt text](https://github.com/OctThe16th/FewShotAuthorClassificationMetaLearning/raw/master/images/AccuracyGuttenberg5Way.PNG)

Using as inputs 1280 tokens worth of text taken from different books from an out of sample author, a 20 way top 1 accuracy of 17% was obtrained on unseen data.
![alt text](https://github.com/OctThe16th/FewShotAuthorClassificationMetaLearning/raw/master/images/AccuracyGuttenberg20Way.PNG)


The discrepancy between the results most likely can be explained by the fact that redditor tend to have very homogeneous
discourse, commenting on a specific topic, which would be easier to classify between.


# Toy example
I got the toy example on sinus regression described in the Maml paper going to double check the algorithm. I could test a few normalizing layer and found that batch norm hurts the result, probably because of the running stats not being properly interpolated, and layer norm helped most.

# Possible uses
Using reptile on NLP problems could be used for effective transfer learning.
More specifically, few shot author recognition could be used to detect author that might wish to stay anonymous, i.e. automatically detecting fake news.
It could also be used for recommendation system, by finding the authors that most could pass for the authors you like.
