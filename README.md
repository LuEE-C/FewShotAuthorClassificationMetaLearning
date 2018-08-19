# FewShotAuthorClassificationMetaLearning
With Reptile https://blog.openai.com/reptile/ I attempt to meta learn few shot author classification. Currently testing the viability on the Gutenberg dataset https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html.

# Toy example
I got the toy example on sinus regression described in the Maml paper going to double check the algorithm. I could test a few normalizing layer and found that batch norm hurts the result, probably because of the running stats not being properly interpolated, and layer norm helped most.

 # Possible uses
 Using reptile on NLP problems could be used for effective transfer learning.
 More specifically, few shot author recognition could be used to detect author that might wish to stay anonymous, i.e. automatically detecting fake news.