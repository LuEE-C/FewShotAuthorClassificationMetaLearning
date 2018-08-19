# FewShotAuthorClassificationMetaLearning
With Reptile https://blog.openai.com/reptile/ I attempt to meta learn few shot author classification. Currently testing the viability on the Gutenberg dataset.

# Toy example
I got the toy example on sinus regression described in the Maml paper going to double check the algorithm. I could test a few normalizing layer and found that batch norm hurts the result, probably because of the running stats not being properly interpolated, and layer norm helped most.