# Deep Structured Semantic Models (DSSM)

This is a repository for IEOR4579 final project, owned by Flora Lin, Lavanda Wang, Weijian Shi


##

## 1. Introduction of DSSM

* Latent Semantic Model
  * Before the introduction of Deep Structured Semantic Models (DSSM), LSM were used to
    map search query to relevant documents at a semantic level.
  * Trained in unsupervised methods and search results were not very ideal
* Deep Structured Semantic Models (DSSM)
  * An extension to existed LSM that incorporates Word Hashing method to reduce
    dimensionality to calculate similarity between texts
  * Trained model using clickthrough data to improve document searching performance
* Milestone paper: ML before, deep learning widely used after

## 2. Word Hashing
| Before this paper:                                                                                            | In this paper:                                                  |
|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| **Used one hot encoding:**                                                                                    | **Word hashing**                                                |
| Given bag of  N words, each word can be represented by categorical variable {0,1}                             | Given a text, we do letter-trigram:                             |
| Given a text, we do one hot encoding:                                                                         | Text → #text# → [#te, tex, ext, xt#], Which is a sliding window algorithm                            |
| “It was the best of times” = [1, 1, 1, 1, 1, 0, 0, 0]<br/>“It was the worst of times” = [1, 1, 1, 0, 1, 1, 0, 0]<br/>“It was the age of wisdom” = [1, 1, 1, 0, 1, 0, 1, 1]<br/>“It was the age of foolishness” = [1, 1, 1, 0, 1, 0, 1, 1] | Greatly reduced the dimensionality of vectors from bag of words                                                             |



## 3. Transform Query to Matrix
Process:
* One hot representation x 
* Weight matrix W * one hot → letter n-gram one hot 
* Embedding matrix E * letter n-gram one hot → vector y 
* Both of W and E are linear transformation so they can be combined: Y = E * W * x (very similar with Word2vec!)


## 4. DSSM Structure
<img src="https://github.com/WeijianShi/DSSM_Research/blob/main/Data/DSSM.jpg">




