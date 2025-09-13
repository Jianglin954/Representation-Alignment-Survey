# Awesome papers in the fields of computer vision, machine learning, pattern recognition, data mining, and natural language processing.


![Paper Reading](https://img.shields.io/badge/PhD-Paper_Reading-green)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![Paper Reading](https://img.shields.io/badge/Fields-CV_ML_PR_DM_NLP-blue)

This is a collection of awesome papers I have read (carefully or roughly) in the fields of computer vision, machine learning, pattern recognition, data mining, and natural language processing (where the notes only represent my personal views). The collection will be continuously updated, so stay tuned. Any suggestions and comments are welcome (jianglinlu@outlook.com). 


## Contents
- [Manifold Learning](#Manifold)
  - [Nonlinear Dimensionality Reduction](#NonlinearDR)
  - [Subspace Learning](#subspaceLearning) 
  - [Graph Construction](#graphConstruction)
- [Sparse Representation](#sparseRepre)
- [Low-Rank Representation](#LRR)
- [Clustering](#clustering)
  - [Shallow Clustering](#shallowClustering)
  - [Deep Clustering](#deepclustering)
- [Learning to Hash](#hashing)
  - [Shallow Hashing](#shallowHash)
  - [Deep Hashing](#deepHash)
- [Domain Adaptation](#DA)
  - [Shallow Domain Adaptation](#shallowDA)
  - [Deep Domain Adaptation](#deepDA)
- [Convolutional Neural Network](#CNN)
- [Neural Tangent Kernel](#NTK)
- [Transformers](#Transformers)
  - [Vision Transformers](#ViT)
  - [Graph Transformers](#GraphTransformers)
- [Graph Neural Network](#GNN)
  - [Spectral-based GNN](#SpectralGNN)
  - [Spatial-based GNN](#SpatialGNN)
  - [Graph Pooling](#GraphPooling)
  - [Graph Structure Learning](#LGL)
  - [GNNs+LMs](#GNNLM)
  - [GNN Pruning](#GNNpruning) 
  - [Self-Supervised GNN](#SSLGNN)
  - [GNN Pre-training](#GNNPreTrain)
  - [GNN Adversarial Attacks](#GNNAA) 
  - [Graph Domain Adaptation](#GraphDomainAdaptation)
  - [Graph Data Augmentation](#GraphDataAugmentation)
  - [Graph Generation](#GraphGeneration)
  - [Causality with Graph](#GraphCausality)
  - [Weisfeiler-Lehman Test](#wltest)
  - [Graph Information Bottleneck](#graphInfoBottle)
  - [Deeper GNN](#deeperGNN)
  - [Graph OOD Generalization](#GraphOOD)
  - [Few-Shot Learning on Graph](#fewshotGNN)
- [Visual Question Answering](#VQA)
- [Implicit Neural Representations](#ImplicitNerualRepresent)
- [Deep Generative Models](#deepGenerativeModels)
  - [Generative Adversarial Networks](#GANs)
  - [Variational Autoencoders](#VAEs)
  - [Normalizing Flows](#normalizingFlows)
  - [Autoregressive Models](#autoregress) 
  - [Diffusion Models](#DiffusionModels)
  - [Score-based Generative Models](#scoreGM)
- [Network Compression](#networkcompression)
  - [Pruning](#pruning)
  - [Knowledge Distillation](#knowDistil)
  - [Network Quantization](#quantization)
  - [Low-Rank Factorization](#LRF)
- [Dataset Distillation](#datasetDistil)
- [Learning with Label Noise](#labelnoise)
  - [Statistically Inconsistent Classifiers](#SICnoise)
  - [Statistically Consistent Classifiers](#SCCnoise)
- [Contrastive Learning](#CLR)
- [Low-Level Vision](#llv)
  - [High Dynamic Range Imaging](#HDR)
  - [Image Super-Resolution](#ImageSR)
  - [Image Low-Light Enhancement](#ImageLLE)
- [Vision Language Pretraining](#VLP)
- [Point Cloud](#pointcloud)
- [Causal Inference](#cause)
- [Natural Language rocessing](#NLP)
- [AI for Sciences](#AI4S)
  - [Drug-Target Binding](#DTA)
- [Others](#others)
  - [Variational Inference](#VI)
  - [Procrustes Problem](#procrustes)
  - [Label Propagation](#labelPropagation)
  - [CUR Decomposition](#cur)
  - [Matrix Completion](#matrixcompletion)
  - [Optimization Methods](#Optimization)
  - [Bilevel Programming](#BiP)
  - [PAC Learning](#PACLearning)
  - [Information Theory](#informationTheory)
  - [Probability Theory](#ProbabilityTheory)
  - [Gumbel-Max Trick](#GumbelmaxTrick)
  - [Stochastic Differential Equations](#SDE)
  - [Quantum Computing](#quantumcomputing)
- [Learning Sources](#learningsources)





<a name="Manifold" />

## Manifold Learning [[Back to Top]](#)



<a name="NonlinearDR" />

### Nonlinear Dimensionality Reduction

1. **A Global Geometric Framework for Nonlinear Dimensionality Reduction.** *Joshua B. Tenenbaum et al, Science 2000.*  [[PDF]](https://www.science.org/doi/10.1126/science.290.5500.2319) [[Author]](http://web.mit.edu/cocosci/josh.html)   
Notes: This is a classical paper that proposes **Isometric Feature Mapping (ISOMAP)** for nonlinear dimensionality reduction, which contains three step including neighborhood graph construction, shortest paths computing, and low-dimensional embedding.

1. **Nonlinear Dimensionality Reduction by Locally Linear Embedding.** *Sam T. Roweis et al, Science 2000.*  [[PDF]](https://www.science.org/doi/10.1126/science.290.5500.2323)  [[Author]](https://cs.nyu.edu/~roweis/)   
Notes: This is a classical paper that proposes **Locally Linear Embedding (LLE)** for nonlinear dimensionality reduction, which, being different from ISOMAP, assumes that each data point and its neighbors lie on or close to a locally linear patch of the manifold. The local geometry of these patches is characterized by linear coefficients that reconstruct each data point from its neighbors. 

1. **Laplacian Eigenmaps for Dimensionality Reduction and Data Representation.** *Mikhail Belkin et al, Neural Computation 2003.*  [[PDF]](https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf) [[Author]](http://misha.belkin-wang.org/)    
Notes: This is a classical paper that proposes **Laplacian Eigenmaps (LE)** for nonlinear dimensionality reduction and data representation, which uses the notion of Laplacian of the graph to compute a low-dimensional representation of the data set that optimally preserves local neighborhood information in a certain sense.

1. **Locality Preserving Projections.** *Xiaofei He et al, NIPS 2003.*  [[PDF]](https://proceedings.neurips.cc/paper/2003/file/d69116f8b0140cdeb1f99a4d5096ffe4-Paper.pdf) [[Author]](http://www.cad.zju.edu.cn/home/xiaofeihe/)   
This paper proposes **Locality Preserving Projections (LPP)**, which computes a linear projection matrix that maps the data point to a subspace. The linear transformation optimally preserves local neighborhood information in a certain sense. This work can be regarded as a linear extension of Laplacian Eigenmaps (LE).

1. **Neighborhood Preserving Embedding.** *Xiaofei He et al, ICCV 2005.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1544858) [[Author]](http://www.cad.zju.edu.cn/home/xiaofeihe/)    
This paper proposes **Neighborhood Preserving Embedding (NPE)**, which aims at preserving the local neighborhood structure on the data manifold. Here, the locality or local structure means that each data point can be represented as a linear combination of its neighbors. This work can be regarded as a linear extension of Locally Linear Embedding (LLE).

1. **Graph Embedding and Extensions: A General Framework for Dimensionality Reduction.** *Shuicheng Yan et al, IEEE TPAMI 2007.*  [[PDF]](https://ieeexplore.ieee.org/document/4016549) [[Author]](https://yanshuicheng.ai/)     
Notes: This paper proposes a general framework called **Graph Embedding** for linear dimensionality reduction, in which an intrinsic graph characterizes the intraclass compactness while a penalty graph characterizes the interclass separability. 


1. **Learning Signal-Agnostic Manifolds of Neural Fields.** *Yilun Du et al, NeurIPS 2021.*  [[PDF]](https://arxiv.org/pdf/2111.06387.pdf) [[Author]](https://yilundu.github.io/)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()

<a name="subspaceLearning" />

### Subspace Learning


1. **Spectral Regression for Efficient Regularized Subspace Learning.** *Deng Cai et al, ICCV 2007.*  [[PDF]](https://ieeexplore.ieee.org/document/4016549) [[Author]](http://www.cad.zju.edu.cn/home/dengcai/)   
Notes: This paper proposes **Spectral Regression (SR)** for subspace learning, which casts the problem of learning the projective functions into a regression framework and avoids the eigen-decomposition of dense matrices. It is worth noting that different kinds of regularizers can be naturally incorporated into SR such as L<sub>1</sub> regularization.


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()






<a name="graphConstruction" />

### Graph Construction

1. **Graph Construction and $b$-Matching for Semi-supervised Learning.** *Tony Jebara et al, ICML 2009.*  [[PDF]](https://icml.cc/Conferences/2009/papers/188.pdf) [[Author]](http://www.cs.columbia.edu/~jebara/)

1. **Influence of Graph Construction on Semi-supervised Learning.** *Celso Sousa et al, ECML PKDD 2013.*  [[PDF]](https://link.springer.com/chapter/10.1007/978-3-642-40994-3_11) [[Author]](https://scholar.google.com/citations?user=i8jXj9kAAAAJ&hl=zh-CN&oi=sra)

1. **How to Learn a Graph from Smooth Signals.** *Vassilis Kalofolias et al, AISTATS 2016.*  [[PDF]](http://proceedings.mlr.press/v51/kalofolias16.pdf) [[Author]](https://scholar.google.ch/citations?user=Bz1RQ8MAAAAJ&hl=en)


1. **A Quest for Structure: Jointly Learning the Graph Structure and Semi-Supervised Classification.** *Xuan Wu et al, CIKM 2018.*  [[PDF]](https://arxiv.org/pdf/1909.12385.pdf) [[Author]]()    
Notes: This paper proposes **Parallel Graph Learning (PG-Learn) for the graph construction step of semi-supervised learning. The two main ingredients include a) a gradient-based optimization of the edge weights (different kernel bandwidths in each dimension) and b) a parallel hyperparameter search algorithm. It adopts LGC algorithm and the corresponding solution can be found without explicitly taking any matrix inverse and instead using the power method. 




1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()














<a name="sparseRepre" />

## Sparse Representation [[Back to Top]](#)
F
1. **Regression Shrinkage and Selection Via the Lasso.** *Rob Tibshirani, Journal of the Royal Statistical Society 1996.*  [[PDF]](https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/j.2517-6161.1996.tb02080.x) [[Author]](http://tibshirani.su.domains/)   
Notes: This is a classical paper that proposes **Least absolute shrinkage and selection operator (LASSO)** for linear regression, which minimizes the residual sum of squares subject to the sum of the absolute value of the coefficients being less than a constant. AKA L<sub>1</sub> penalty. 

1. **Regularization and Variable Selection via the Elastic Net.** *Hui Zou et al, Journal of the royal statistical society 2005.*  [[PDF]](https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/j.1467-9868.2005.00503.x) [[Author]](http://users.stat.umn.edu/~zouxx019/)   
Notes: This paper proposes **Elastic Net** for regularization and variable selection, which encourages a grouping effect where strongly correlated predictors tend to be in or out of the model together. The Elastic Net combines L<sub>2</sub> regularization and L<sub>1</sub> regularization together, and can be viewed as a generalization of LASSO.

1. **Sparse Principal Component Analysis.** *Hui Zou et al, Journal of Computational and Graphical Statistics 2006.*  [[PDF]](https://hastie.su.domains/Papers/spc_jcgs.pdf) [[Author]](http://users.stat.umn.edu/~zouxx019/)   
Notes: This paper proposes **Sparse Principal Component Analysis (SPCA)** that introduces the LASSO or Elastic Net into Principal Component Analysis (PCA) to produce modified principal components with sparse loadings. It formulates PCA as a regression-type optimization problem and then obtains sparse loadings by imposing the LASSO or Elastic Net constraint on the regression coefficients. The Theorem 4 of **Reduced Rank Procrustes Rotation** is useful. 

1. **Robust Face Recognition via Sparse Representation.** *John Wright et al, IEEE TPAMI 2009.*  [[PDF]](https://ieeexplore.ieee.org/document/4483511) [[Author]](http://www.columbia.edu/~jw2966/)   
Notes:

1. **Robust principal component analysis?.** *Emmanuel J. Cand`es et al, Journal of the ACM 2011.*  [[PDF]](https://dl.acm.org/doi/abs/10.1145/1970392.1970395) [[Author]](https://candes.su.domains/)   
Notes:


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




### Resources

1. **SLEP: Sparse Learning with Efficient Projections.** *Jun Liu et al, Arizona State University 2009.*  [[PDF]](https://media.gradebuddy.com/documents/411659/9e8ca2d6-1223-47a2-81b1-02c81e2f40ce.pdf) [[Resource]](http://www.yelabs.net/software/SLEP/) [[Author]](https://sites.google.com/site/junliupage/)  
Notes: This paper develops a **Sparse Learning with Efficient Projections (SLEP)** package written in Matlab for sparse representation learning.











































<a name="LRR" />

## Low-Rank Representation [[Back to Top]](#)

1. **Robust Subspace Segmentation by Low-Rank Representation.** *Guangcan Liu et al, ICML 2010.*  [[PDF]](https://icml.cc/Conferences/2010/papers/521.pdf) [[Author]](https://sites.google.com/site/guangcanliu/)   
Notes:

1. **Robust Recovery of Subspace Structures by Low-Rank Representation.** *Guangcan Liu et al, IEEE TPAMI 2013.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6180173) [[Author]](https://sites.google.com/site/guangcanliu/)   
Notes:

1. **The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices.** *Zhouchen Lin et al, arXiv 2013.*  [[PDF]](https://arxiv.org/abs/1009.5055) [[Author]](https://zhouchenlin.github.io/)   
Notes:



1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()













































<a name="clustering" />

## Clustering [[Back to Top]](#)


<a name="shallowClustering" />

### Shallow Clustering

1. **Large Scale Spectral Clustering with Landmark-Based Representation.** *Xinlei Chen et al, AAAI 2011.*  [[PDF]](http://www.cad.zju.edu.cn/home/dengcai/Publication/Conference/2011_AAAI-LSC.pdf) [[Author]](https://xinleic.xyz/)   
Notes: This paper adopts an **Anchor Graph** for spectral clustering.

1. **Sparse Subspace Clustering: Algorithm, Theory, and Applications.** *Ehsan Elhamifar et al, IEEE TPAMI 2013.*  [[PDF]](https://ieeexplore.ieee.org/document/6482137) [[Author]](https://khoury.northeastern.edu/home/eelhami/)   
Notes: This papers proposes **Sparse Subspace Clustering (SSC)** which introduces sparse representation into the subspace clustering problem, and define the **Self-Expressiveness** property: each data point in a union of subspaces can be efficiently reconstructed by a combination of other points in the dataset.

1. **Clustering and Projected Clustering with Adaptive Neighbors.** *Feiping Nie et al, KDD 2014.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/2623330.2623726) [[Author]](https://sites.google.com/site/feipingnie/)  
Notes: This paper proposes **Clustering with Adaptive Neighbors (CAN)** to learn the data similarity matrix and clustering structure simultaneously. It is worth noting that they present an effective method to determine the regularization parameter considering the locality of the data.


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()





<a name="deepclustering" />

### Deep Clustering

1. **Deep Subspace Clustering Networks.** *Pan Ji et al, NIPS 2017.*  [[PDF]](https://arxiv.org/abs/1709.02508) [[Author]](https://sites.google.com/view/panji530)   
Notes: This is the first deep subspace clustering network, however, it has been proved to be ill-posed by the [paper](https://openreview.net/forum?id=FOyuZ26emy).


1. **A Critique of Self-Expressive Deep Subspace Clustering.** *Benjamin David Haeffele et al, ICLR 2021.*  [[PDF]](https://openreview.net/forum?id=FOyuZ26emy) [[Author]](https://www.cis.jhu.edu/~haeffele/)   
Notes: This papers show that many previous deep subspace networks are ill-posed, and their performance improvement is largely attributable to an ad-hoc post-processing step.

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()












<a name="hashing" />

## Learning to Hash [[Back to Top]](#)


<a name="shallowHash" />

### Shallow Hashing

1. **Locality-Sensitive Hashing Scheme based on p-Stable Distributions.** *Mayur Datar et al, SCG 2004.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/997817.997857) [[Author]](https://www.linkedin.com/in/mayur-datar-b0a65018/)   
Notes: This paper proposes a novel **Locality-Sensitive Hashing (LSH)** for the Approximate Nearest Neighbor Problem under L<sub>p</sub> norm, based on p-stable distributions.  The key idea is to hash the points using several hash functions so as to ensure that, for each function, the probability of collision is much higher for objects which are close to each other than for those which are far apart. Then, one can determine near neighbors by hashing the query point and retrieving elements stored in buckets containing that point.

1. **Spectral Hashing.** *Yair Weiss et al, NIPS 2008.*  [[PDF]](https://proceedings.neurips.cc/paper/2008/file/d58072be2820e8682c0a27c0518e805e-Paper.pdf) [[Author]](https://www.cs.huji.ac.il/~yweiss/)   
Notes: This paper proposes **Spectral Hashing (SH)** where the bits are calculated by thresholding a subset of eigenvectors of the Laplacian of the similarity graph. The basic idea is to embed the data in a Hamming space such that the neighbors in the original data space remain neighbors in the Hamming space. 

1. **Hashing with Graphs.** *Wei Liu et al, ICML 2011.*  [[PDF]](https://icml.cc/2011/papers/6_icmlpaper.pdf) [[Author]](https://sites.google.com/view/cuweiliu)   
Notes: This paper proposes **Anchor Graph Hashing (AGH)** which builds an approximate neighborhood graph using Anchor Graphs, resulting in O(n) time for graph construction. The graph is sufficiently sparse with performance approaching to the true KNN graph as the number of anchors increases.

1. **Iterative Quantization: A Procrustean Approach to Learning Binary Codes for Large-Scale Image Retrieval.** *Yunchao Gong et al, IEEE TPAMI 2013.*  [[PDF]](https://ieeexplore.ieee.org/document/6296665) [[Author]](https://www.linkedin.com/in/yunchao-gong-150a32a/)   
Notes: This paper proposes **Iterative Quantization (ITQ)** that finds a rotation of zero-centered data so as to minimize the quantization error of mapping this data to the vertices of a zero-centered binary hypercube. The optimization problem is intrinsically an [Orthogonal Procrustes problem](https://web.stanford.edu/class/cs273/refs/procrustes.pdf).

1. **The Power of Asymmetry in Binary Hashing.** *Behnam Neyshabur et al, NIPS 2013.*  [[PDF]](https://proceedings.neurips.cc/paper/2013/file/84438b7aae55a0638073ef798e50b4ef-Paper.pdf) [[Author]](https://www.neyshabur.net/)   
Notes: This paper proves that shorter and more accurate hash codes can be obtained by using two distinct code maps. The asymmetry here is defined in function space, i.e., adopting different hashing functions for similarity calculation.

1. **Sparse Projections for High-dimensional Binary Codes.** *Yan Xia et al, CVPR 2015.*  [[PDF]](https://ieeexplore.ieee.org/document/7298954) [[Author]](https://www.linkedin.com/in/xiayan-ustc/)   
Notes: This paper proposes **Sparse Projections (SP)** which imposes orthogonal and L<sub>0</sub> constraints on the hashing projection matrix. For the resulting non-convex optimization problem, they adopt the variable-splitting and penalty techniques.

1. **Supervised Discrete Hashing.** *Fumin Shen et al, CVPR 2015.*  [[PDF]](https://ieeexplore.ieee.org/document/7298598) [[Author]](https://scholar.google.com.au/citations?user=oqYL6fQAAAAJ&hl=en)  
Notes: This paper proposes **Supervised Discrete Hashing (SDH)** that solves the discrete optimization without any relaxations using a discrete cyclic coordinate descent (DCC) algorithm. The assumption is that good hash codes are optimal for linear classification. 

1. **Fast Supervised Discrete Hashing.** *Jie Gui et al, TPAMI 2018.*  [[PDF]](https://ieeexplore.ieee.org/document/7873258) [[Author]](https://guijiejie.github.io/)   
Notes: This paper proposes **Fast Supervised Discrete Hashing (FSDH)** that regress the class labels of training data to the corresponding hash codes to accelerate the SDH. It avoids iterative hash code-solving step of the DCC algorithm. 



<a name="deepHash" />

### Deep Hashing

1. **Feature Learning based Deep Supervised Hashing with Pairwise Labels.** *Wu-Jun Li et al, IJCAI 2016.*  [[PDF]](https://www.ijcai.org/Proceedings/16/Papers/245.pdf) [[Author]](https://cs.nju.edu.cn/lwj/index.htm)  
Notes: This paper proposes the first deep hashing method called **Deep Pairwise-Supervised Hashing (DPSH)** for applications with pairwise labels, which can perform simultaneous feature learning and hash-code learning. This paper can be regarded as a deep learning extension of [Latent Factor Hashing (LFH)](https://dl.acm.org/doi/pdf/10.1145/2600428.2609600).

1. **Asymmetric Deep Supervised Hashing.** *Qing-Yuan Jiang et al, AAAI 2018.*  [[PDF]](https://arxiv.org/abs/1707.08325) [[Author]](https://jiangqy.github.io/)   
Notes: This paper proposes the first asymmetric deep hashing method called **Asymmetric Deep Supervised Hashing(ADSH)**, which can treats query points and database points in an asymmetric way. Specifically, ADSH learns a deep hash function only for query points while the hash codes for
database points are directly learned.

1. **Deep Supervised Hashing with Anchor Graph.** *Yudong Chen et al, ICCV 2019.*  [[PDF]](https://ieeexplore.ieee.org/document/9010953) [[Author]]()    
Notes: This paper proposes **Deep Anchor Graph Hashing (DAGH)**, which adopts an **Anchor Graph** to learn the hash codes of the whole training samples directly during training. Since in different epochs the anchors used are different, the entire training samples will be trained if given enough epochs. This paper can also be regarded as an asymmetric deep hashing method. 

1. **Deep Cross-Modal Hashing.** *Qing-Yuan Jiang et al, CVPR 2017.*  [[PDF]](https://ieeexplore.ieee.org/document/9010953) [[Author]](https://jiangqy.github.io/)   
Notes: This paper proposes the first deep cross-modal hashing called **Deep Cross-Modal Hashing (DCMH)** which can be regarded as a cross-modal extension of [Deep Pairwise-Supervised Hashing (DPSH)](https://www.ijcai.org/Proceedings/16/Papers/245.pdf).


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




### Survey

1. **A Survey on Learning to Hash.** *Jingdong Wang et al, IEEE TPAMI 2018.*  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7915742) [[Author]](https://jingdongwang2017.github.io/) 

[//]: 1. **A Survey on Deep Hashing Methods.** *Xiao Luo et al, ACM TKDD 2022.*  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7915742) [[Author]]() 












































<a name="DA" />

## Domain Adaptation [[Back to Top]](#)

1. **My Personal Learning Notes on Domain Adaptation.** *Jianglin Lu.*  [[PDF]](https://jianglin954.github.io/Awesome-papers-in-the-fileds-of-CV-ML-PR-DM/files/Introduction_of_Two_Domain_Adaptation_Methods.pdf)  



<a name="shallowDA" />

### Shallow Domain Adaptation

1. **Domain Adaptation under Target and Conditional Shift.** *Kun Zhang et al, ICML 2013.*  [[PDF]](http://proceedings.mlr.press/v28/zhang13d.pdf) [[Author]](https://www.andrew.cmu.edu/user/kunz1/index.html)    
Notes: This paper exploits importance reweighting or sample transformation to find the learning machine that works well on test data, and propose to estimate the weights or transformations by reweighting or transforming training data to reproduce the covariate distribution on the test domain.

1. **Domain Adaptation with Conditional Transferable Components.** *Mingming Gong et al, ICML 2016.*  [[PDF]](http://proceedings.mlr.press/v48/gong16.pdf) [[Author]](https://mingming-gong.github.io/)    
Notes:



1. **Structural Re-weighting Improves Graph Domain Adaptation.** *Shikun Liu et al, ICML 2023.*  [[PDF]](https://arxiv.org/pdf/2306.03221.pdf) [[Author]]()




<a name="deepDA" />

### Deep Domain Adaptation

1. **Universal Domain Adaptation.** *Kaichao You et al, CVPR 2019.*  [[PDF]](https://ieeexplore.ieee.org/document/8954135) [[Author]](https://youkaichao.github.io/about)   
Notes: This paper introduces **Universal Domain Adaptation (UDA)** that requires no prior knowledge on the label sets of source and target domains.


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()





### Survey

1. **A Survey on Transfer Learning.** *Sinno Jialin Pan et al, IEEE TKDE 2010.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5288526) [[Author]](http://www.cse.cuhk.edu.hk/~sinnopan/) 



1. **A Comprehensive Survey on Transfer Learning.** *Fuzhen Zhuang et al, Proceedings of the IEEE 2021.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9134370) [[Author]](https://fuzhenzhuang.github.io/index.html) 


























<a name="NTK" />

## Neural Tangent Kernel [[Back to Top]](#)


1. **Neural Tangent Kernel: Convergence and Generalization in Neural Networks.** *Arthur Jacot et al, NeurIPS 2018.*  [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf) [[Author]](https://sites.google.com/view/arthurjacot/)


1. **Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent.** *Jaehoon Lee et al, NeurIPS 2019.*  [[PDF]](https://arxiv.org/pdf/1902.06720.pdf) [[Author]](https://jaehlee.github.io/)


1. **Graph Neural Tangent Kernel: Fusing Graph Neural Networks with Graph Kernels.** *Simon S. Du et al, NeurIPS 2019.*  [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2019/file/663fd3c5144fd10bd5ca6611a9a5b92d-Paper.pdf) [[Author]](https://simonshaoleidu.com/)


1. **Gradient Descent Finds Global Minima of Deep Neural Networks.** *Simon S. Du et al, ICML 2019.*  [[PDF]](http://proceedings.mlr.press/v97/du19c/du19c.pdf) [[Author]](https://simonshaoleidu.com/)



1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()












<a name="CNN" />

## Convolutional Neural Network [[Back to Top]](#)

1. **Going deeper with convolutions.** *Christian Szegedy et al, CVPR 2015.*  [[PDF]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) [[Author]](https://scholar.google.com/citations?user=3QeF7mAAAAAJ&hl=zh-CN&oi=sra)

1. **Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.** *Kaiming He et al, ICCV 2015.*  [[PDF]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf) [[Author]](https://kaiminghe.github.io/)

1. **Deep Residual Learning for Image Recognition.** *Kaiming He et al, CVPR 2016.*  [[PDF]](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) [[Author]](https://kaiminghe.github.io/)

1. **Searching for Activation Functions.** *Prajit Ramachandran et al, arXiv 2017.*  [[PDF]](https://arxiv.org/pdf/1710.05941.pdf) [[Author]](https://scholar.google.com/citations?user=ktKXDuMAAAAJ&hl=en)


1. **Densely Connected Convolutional Networks.** *Gao Huang et al, CVPR 2017.*  [[PDF]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) [[Author]](http://www.gaohuang.net/)

1. **A ConvNet for the 2020s.** *Zhuang Liu et al, CVPR 2022.*  [[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf) [[Author]](https://liuzhuang13.github.io/)



1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




























<a name="Transformers" />

## Transformers [[Back to Top]](#)





<a name="ViT" />

### Vision Transformers


1. **Attention Is All You Need.** *Ashish Vaswani et al, NIPS 2017.*  [[PDF]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Author]](https://scholar.google.com/citations?user=oR9sCGYAAAAJ&hl=en&oi=sra)



1. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.** *Ze Liu et al, ICCV 2021.*  [[PDF]](https://ieeexplore.ieee.org/document/9710580) [[Author]](https://zeliu98.github.io/)

1. **Tokens-to-Token ViT: Training Vision Transformers From Scratch on ImageNet.** *Li Yuan et al, ICCV 2021.*  [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_Tokens-to-Token_ViT_Training_Vision_Transformers_From_Scratch_on_ImageNet_ICCV_2021_paper.pdf) [[Author]](https://yuanli2333.github.io/)



1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()


<a name="GraphTransformers" />

### Graph Transformers

1. **Graph Transformer Networks.** *Seongjun Yun et al, NeurIPS 2019.*  [[PDF]](https://arxiv.org/abs/1911.06455) [[Author]](https://scholar.google.com/citations?user=8-MZ2RwAAAAJ&hl=en)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()


1. **Meta-Transformer: A Unified Framework for Multimodal Learning.** *Yiyuan Zhang et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2307.10802.pdf) [[Author]](https://invictus717.github.io/)



### Survey

1. **A Survey on Vision Transformer.** *Kai Han et al, IEEE TPAMI 2022.*  [[PDF]](https://ieeexplore.ieee.org/document/9716741) [[Author]](https://scholar.google.com/citations?user=vThoBVcAAAAJ&hl=en&oi=sra)




































<a name="GNN" />

## Graph Neural Network [[Back to Top]](#)


1. **My Personal Learning Notes on Graph Neural Network.** *Jianglin Lu.* [[PDF]](https://jianglin954.github.io/Awesome-papers-in-the-fileds-of-CV-ML-PR-DM/files/Introduction_of_GNN.pdf) 



<a name="SpectralGNN" />

### Spectral-based GNN

1. **Simplifying Graph Convolutional Networks.** *Felix Wu et al, ICML 2019.*  [[PDF]](http://proceedings.mlr.press/v97/wu19e/wu19e.pdf) [[Author]](https://sites.google.com/view/felixwu/home)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()
1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()






<a name="SpatialGNN" />

### Spatial-based GNN


1. **Semi-Supervised Classification with Graph Convolutional Networks.** *Thomas N. Kipf et al, ICLR 2017.*  [[PDF]](https://arxiv.org/pdf/1609.02907.pdf) [[Author]](https://tkipf.github.io/) [[Code]](https://github.com/tkipf/pygcn)





1. **Inductive Representation Learning on Large Graphs.** *William L. Hamilton et al, NeurIPS 2017.*  [[PDF]](https://proceedings.neurips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf) [[Author]](https://www.cs.mcgill.ca/~wlh/) [[Code]](https://github.com/williamleif/graphsage-simple/)   
Notes: Most existing approaches are inherently transductive, which require all nodes in the graph are present during training of the embeddings. This paper proposes **SAmple and aggreGatE (GraphSAGE)** for inductive node embedding, which generates embeddings by sampling and aggregating features from a node's local neighborhood. 

1. **Graph Attention Networks.** *Petar Veličković et al, ICLR 2018.*  [[PDF]](https://arxiv.org/abs/1710.10903) [[Author]](https://petar-v.com/)   
 

1. **Pitfalls of Graph Neural Network Evaluation.** *Oleksandr Shchur et al, NeurIPS 2018.*  [[PDF]](https://arxiv.org/pdf/1811.05868.pdf) [[Author]](https://shchur.github.io/)





1. **Deeper Insights Into Graph Convolutional Networks for Semi-Supervised Learning.** *Qimai Li et al, AAAI 2018.*  [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/11604) [[Author]](https://liqimai.github.io/)


1. **Adaptive Sampling Towards Fast Graph Representation Learning.** *Wenbing Huang et al, NeurIPS 2018.* [[PDF]](https://proceedings.neurips.cc/paper/2018/file/01eee509ee2f68dc6014898c309e86bf-Paper.pdf) [[Author]](https://gsai.ruc.edu.cn/addons/teacher/index/info.html?user_id=31&ruccode=ADIIMVRnBzFXMFdnVTAIOw%3D%3D&ln=en) 


1. **LanczosNet: Multi-Scale Deep Graph Convolutional Networks.** *Renjie Liao et al, ICLR 2019.*  [[PDF]](https://openreview.net/pdf?id=BkedznAqKQ) [[Author]](https://lrjconan.github.io/) [[Code]](https://github.com/lrjconan/LanczosNetwork) 

1. **Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition.** *Ziyu Liu et al, CVPR 2020.*  [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Disentangling_and_Unifying_Graph_Convolutions_for_Skeleton-Based_Action_Recognition_CVPR_2020_paper.pdf) [[Author]](https://kenziyuliu.github.io/) [[Code]](https://github.com/kenziyuliu/ms-g3d)



1. **Grale: Designing Networks for Graph Learning.** *Jonathan Halcrow et al, KDD 2020.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403302) [[Author]](https://scholar.google.com/citations?user=2zZucy4AAAAJ&hl=en&oi=ao)   


1. **Graph Neural Networks with Adaptive Residual.** *Xiaorui Liu et al, NeurIPS 2021.*  [[PDF]](https://proceedings.neurips.cc/paper/2021/file/50abc3e730e36b387ca8e02c26dc0a22-Paper.pdf) [[Author]]()


1. **E(n) Equivariant Graph Neural Networks.** *Victor Garcia Satorras et al, ICML 2021.*  [[PDF]](http://proceedings.mlr.press/v139/satorras21a/satorras21a.pdf) [[Author]](https://scholar.google.com/citations?user=FPRvtUEAAAAJ&hl=zh-CN&oi=sra)


1. **Understanding over-squashing and bottlenecks on graphs via curvature.** *Jake Topping et al, ICLR 2022.*  [[PDF]](https://openreview.net/pdf?id=7UmjRGzp-A) [[Author]](https://scholar.google.com/citations?hl=en&user=kBDis8EAAAAJ)






1. **DropMessage: Unifying Random Dropping for Graph Neural Networks.** *Taoran Fang et al, AAAI 2023.*  [[PDF]](https://arxiv.org/pdf/2204.10037.pdf) [[Author]]()  [[Code]](https://github.com/zjunet/DropMessage)



1. **Graph Neural Networks can Recover the Hidden Features Solely from the Graph Structure.** *Ryoma Sato et al, ICML 2023.*  [[PDF]](https://arxiv.org/pdf/2301.10956.pdf) [[Author]](https://joisino.net/en/)



1. **Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs.** *Chenxiao Yang et al, ICLR 2023.*  [[PDF]](https://arxiv.org/pdf/2212.09034.pdf) [[Author]](https://scholar.google.com/citations?hl=en&user=HScmvrAAAAAJ&view_op=list_works) [[Code]](https://github.com/chr26195/PMLP)










1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()

<a name="GraphPooling" />

### Graph Pooling

1. **Hierarchical Graph Representation Learning with Differentiable Pooling.** *Rex Ying et al, NeurIPS 2018.*  [[PDF]](https://proceedings.neurips.cc/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf) [[Author]](https://cs.stanford.edu/people/rexy/)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()





<a name="LGL" />

### Graph Structure Learning (AKA, Latent Graph Learning)

1. **Deep Convolutional Networks on Graph-Structured Data.** *Mikael Henaff et al, arXiv 2015.*  [[PDF]](https://arxiv.org/pdf/1506.05163.pdf) [[Author]](http://www.mikaelhenaff.net/)   
Notes: 


1. **Learning Conditioned Graph Structures for Interpretable Visual Question Answering.** *Will Norcliffe-Brown et al, .*  [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2018/file/4aeae10ea1c6433c926cdfa558d31134-Paper.pdf) [[Author]](https://www.linkedin.com/in/william-norcliffe-brown/?originalSubdomain=uk)


1. **Adaptive Graph Convolutional Neural Networks.** *Ruoyu Li et al, AAAI 2018.* [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/11691) [[Author]](https://scholar.google.co.uk/citations?user=S23gEPsAAAAJ&hl=en)  
Notes: The bottlenecks of current graph CNNS: a. restrict graph degree; b. required identical graph structure shared among inputs; C. fixed graph constructed without training; d. incapability of learning from topological structure. This paper proposes **Adaptive Graph Convolution Network (AGCN)** that feeds on original data of diverse graph structures. AGCN seems to be designed primarily for graph classification. Besides, AGCN needs an initial graph and suffers from the limitation of transductive models as described in [DGM](https://ieeexplore.ieee.org/document/9763421).

1. **Topology Optimization based Graph Convolutional Network.** *Liang Yang et al, IJCAI 2019.*  [[PDF]](https://www.ijcai.org/proceedings/2019/0563.pdf) [[Author]](http://yangliang.github.io/)   
Notes: This paper proposes **Topology Optimization based GCN (TO-GCN)** to jointly learn the network topology and the parameters of fully connected network. The refinement of the network topology is modeled as a **[Label Propagation](#labelPropagation)** process where the network topology is modeled as the multiplication of the predicted label matrix with its transpose matrix. The TO-GCN also penalizes the high similarities between the nodes from different classes. 

    
1. **Semi-Supervised Learning With Graph Learning-Convolutional Networks.** *Bo Jiang et al, CVPR 2019.*  [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Jiang_Semi-Supervised_Learning_With_Graph_Learning-Convolutional_Networks_CVPR_2019_paper.pdf) [[Author]](https://scholar.google.com/citations?user=n-aTwuMAAAAJ&hl=zh-CN) [[Code]](https://github.com/jiangboahu/GLCN-tf)   
Notes: This paper proposes **Graph Learning-Convolutional Network (GLCN)** for semi-supervised task, which integrates both graph learning and graph convolution in a unified network architecture such that both given and estimated labels are incorporated to provide weakly supervised information for graph structure refinement. The graph learning function is similar to [GAT](https://arxiv.org/abs/1710.10903) and the graph learning loss is similar to [CAN](https://dl.acm.org/doi/pdf/10.1145/2623330.2623726). The graph learned in the sense of probability is dense and lack sparse structure.

1. **Large Scale Graph Learning from Smooth Signals.** *Vassilis Kalofolias et al, ICLR 2019.*  [[PDF]](https://arxiv.org/pdf/1710.05654.pdf) [[Author]](https://scholar.google.ch/citations?user=Bz1RQ8MAAAAJ&hl=en)  
Notes: This papers uses approximate nearest neighbor techniques for large scale graph learning from smooth signals. Also refer to [paper](http://proceedings.mlr.press/v51/kalofolias16.pdf).  

1. **Learning Discrete Structures for Graph Neural Networks.** *Luca Franceschi et al, ICML 2019.*  [[PDF]](http://proceedings.mlr.press/v97/franceschi19a/franceschi19a.pdf) [[Author]](https://scholar.google.com/citations?user=NCls8VMAAAAJ&hl=en&oi=ao) [[Code]](https://github.com/lucfra/LDS-GNN)   
Notes: This paper proposes **Learning Discrete Structures (LDS)** to learn the graph structure and the parameters of GCNs by approximately solving a bilevel program that learns a discrete probability distribution of the edges of the graph. Given two objective functions $F$ and $L$, the outer and inner objectives, and two sets of variables, $\theta \in \mathcal{R}^{m}$ and $\omega \in \mathcal{R}^{d}$, the outer and inner variables, a **Bilevel Program** is given by: $\min_{\theta, \omega_{\theta}}F(\omega_{\theta}, \theta)$ such that $\omega_{\theta} \in \arg \min_{\omega} L(\omega, \theta)$. LDS only works in the transductive setting and the graph topology learned cannot be controlled due to the sampling strategy. 

1. **Graph Structure Learning for Robust Graph Neural Networks** *Wei Jin et al, KDD 2020.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403049) [[Author]](http://cse.msu.edu/~jinwei2/) [[Code]](https://github.com/ChandlerBang/Pro-GNN)     
Notes: This paper proposes **Property GNN (Pro-GNN)** that explores graph properties of sparsity, low rank and feature smoothness to defend adversarial attacks. Pro-GNN simultaneously learns the clean graph structure from perturbed graph and GNN parameters to defend against adversarial attacks. This paper assumes that the graph structure has already been perturbed before training GNNs while the node features are not changed.

1. **Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings.** *Yu Chen et al, NeurIPS 2020.* [[PDF]](https://proceedings.neurips.cc/paper/2020/file/e05c7ba4e087beea9410929698dc41a6-Paper.pdf) [[Author]](http://academic.hugochan.net/) [[Code]](https://github.com/hugochan/IDGL)  
Notes: This paper proposes **Iterative Deep Graph Learning (IDGL)** that learns graph structure and graph embedding simultaneously. The graph learning problem is transferred as a similarity metric learning problem and an adaptive graph regularization is leveraged (assume that the optimized graph structure is potentially a shift from the initial graph structure). IDGL adopts multi-head self-attention with $\epsilon$-neighborhood sparsification for graph construction. An **Anchor Graph** based version is also proposed and the corresponding node-anchor message passing strategy is provided. IDGL works on (semi-)supervised tasks and needs an initial $kNN$ graph construction.

1. **Latent-Graph Learning for Disease Prediction.** *Luca Cosmo et al, MICCAI 2020.*  [[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-59713-9_62) [[Author]](https://www.dsi.unive.it/~cosmo/)   
Notes: This paper proposes an end-to-end trainable graph learning architecture that automatically learns to predict an underlying patients-graph. The edge weight is learned through a sigmoid-like function with two trainable parameters. This method can work in inductive setting since it does not directly optimize a graph for a given population but rather learn a function that predicts the graph from input features. The graph learned is directly used only in a classification loss without any regularization. Besides, the global threshold and the Euclidean space embedding may not be necessarily optimal. 

1. **Graph-Revised Convolutional Network.** *Donghan Yu et al, ECML PKDD 2020.*  [[PDF]](https://arxiv.org/pdf/1911.07123.pdf) [[Author]](https://plusross.github.io/)  [[Code]](https://github.com/PlusRoss/GRCN)   
Notes: This paper proposes **Graph-Revised Convolutional Network (GRCN)**, where a GCN-based graph revision module is introduced for predicting missing edges and revising edge weights w.r.t. downstream tasks via joint optimization. The similarity graph is calculated based on node embedding using certain kernel function (specifically, using dot product in their implementation for simplicity). The **[Representer Theorem](https://alex.smola.org/papers/2001/SchHerSmo01.pdf)** is provided to show that, under certain conditions, the optimal regression function can be expressed as a linear combination of kernel functions defined on training samples. Compared with the graph revision in [GAT](https://arxiv.org/abs/1710.10903) and [GLCN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Jiang_Semi-Supervised_Learning_With_Graph_Learning-Convolutional_Networks_CVPR_2019_paper.pdf) which use entrywise product, GRCN adopts the entrywise addition operator in order for new edges to be considered. A graph sparsification process is also proposed and the gradients will only backpropagate through the top-$K$ values. In GRCN, an initial graph is required.   


1. **SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks.** *Bahare Fatemi et al, NeurIPS 2021.* [[PDF]](https://proceedings.neurips.cc/paper/2021/file/bf499a12e998d178afd964adf64a60cb-Paper.pdf) [[Author]](https://baharefatemi.github.io/homepage/) [[Code]](https://github.com/BorealisAI/SLAPS-GNN)   
Notes: This paper proposes **Simultaneous Learning of Adjacency and GNN Parameters with Self-supervision (SLAPS)** for semi-supervised classification, which provides more supervision for inferring a graph structure through self-supervision. The authors also identify a **Supervision Starvation** problem in latent graph learning: the edges between pairs of nodes that are far from labeled nodes receive insufficient supervision. To solve this, a multi-task learning framework is designed by supplementing the classification task with a self-supervised task (which is based on the hypothesis that a graph structure is suitable for predicting the node feature is also suitable for predicting the node labels). Also refer to [paper](http://proceedings.mlr.press/v119/you20a/you20a.pdf).



1. **Graph Structure Estimation Neural Networks.** *Ruijia Wang et al, WWW 2021.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3442381.3449952) [[Author]](https://scholar.google.com/citations?user=DpsuBrsAAAAJ&hl=zh-CN&oi=sra) [[Code]](https://github.com/BUPT-GAMMA/Graph-Structure-Estimation-Neural-Networks)








1. **Graph Structure Learning with Variational Information Bottleneck.** *Qingyun Sun et al, AAAI 2022.*  [[PDF]](https://arxiv.org/pdf/2112.08903.pdf) [[Author]](https://sunqysunqy.github.io/)   
Notes: This paper proposes **Variational Information Bottleneck guided Graph Structure Learning (VIB-GSL)** that advances the **Information Bottleneck** principle for graph structure learning. Refer to [paper](https://proceedings.neurips.cc/paper/2020/file/ebc2aa04e75e3caabda543a1317160c0-Paper.pdf).


1. **Ada-NETS: Face Clustering via Adaptive Neighbour Discovery in the Structure Space.** *Yaohua Wang et al, ICLR 2022.*  [[PDF]](https://arxiv.org/pdf/2202.03800.pdf) [[Author]](https://scholar.google.com/citations?user=TRAwmsgAAAAJ&hl=zh-CN)   
Notes: This paper proposes **Ada-NETS** for face clustering, in which each face is transformed to a new structure space and the similarity is calculated by a weighted combination between cosine similarity and Jaccard similarity. An adaptive neighbor discovery strategy based on **$F_{\beta}$-score** is also proposed to determine a proper number of edges connecting to each face image. 


1. **Robust Graph Structure Learning via Multiple Statistical Tests.** *Yaohua Wang et al, NeurIPS 2022.*  [[PDF]](https://arxiv.org/pdf/2210.03956.pdf) [[Author]](https://scholar.google.com/citations?user=TRAwmsgAAAAJ&hl=zh-CN)   
Notes: This paper proposes views the feature vector of each node as an independent sample, and make the decision of whether creating an edge between two nodes based on their similarity in feature representation by a single statistical test. Transformers based method is proposed, which contains the fourth-order statistics of features. 


1. **pyGSL: A Graph Structure Learning Toolkit.** *Max Wasserman et al, NeurIPS Workshop 2022.*  [[PDF]](https://arxiv.org/pdf/2211.03583.pdf) [[Author]](https://github.com/maxwass) [[Code]](https://github.com/maxwass/pyGSL)   
Notes: This paper introduce **pyGSL**, a Python library that provides efficient implementations of state-of-the-art graph structure learning models along with diverse datasets to evaluate them on. The resource is limited and the code repository is not well-developed.


1. **Learning Continuous Graph Structure with Bilevel Programming for Graph Neural Networks.** *Minyang Hu et al, IJCAI 2022.*  [[PDF]](https://www.ijcai.org/proceedings/2022/0424.pdf) [[Author]](https://scholar.google.com/citations?user=6Saa1ugAAAAJ&hl=zh-CN&oi=sra)   
Notes: This paper proposes to directly model the continuous graph structure with dual-normalization (using a symmetric normalization function), which implicitly imposes sparse constraint and reduces the influence of noisy edges. The whole learning process is formulated as a bilevel programming problem similar to [Learning Discrete Structures](http://proceedings.mlr.press/v97/franceschi19a/franceschi19a.pdf) but armed with an improved **Neumann-IFT** algorithm for optimization. 


1. **GPN: A Joint Structural Learning Framework for Graph Neural Networks.** *Qianggang Ding et al, AAAI 2022.*  [[PDF]](https://arxiv.org/pdf/2205.05964.pdf) [[Author]](http://mrdqg.com/)   
Notes: 


1. **Towards Unsupervised Deep Graph Structure Learning.** *Yixin Liu et al, WWW 2022.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3485447.3512186) [[Author]](https://yixinliu233.github.io/) [[Code]](https://github.com/GRAND-Lab/SUBLIME)   
Notes: This paper proposes an unsupervised learning paradigm for graph structure learning. The proposed method is highly related to **SLAPS** ([paper](https://proceedings.neurips.cc/paper/2021/file/bf499a12e998d178afd964adf64a60cb-Paper.pdf)). In SLAPS, the authors find that "*Although SLAPS2s does not use the node labels in learning an adjacency matrix, it outperforms kNN-GCN (8.4% improvement when using an FP generator). With an FP generator, SLAPS2s even achieves competitive performance with SLAPS; this is mainly because FP does not leverage the supervision provided by GCNC toward learning generalizable patterns that can be used for nodes other than those in the training set*."




1. **Learning Graph Structure from Convolutional Mixtures.** *Max Wasserman et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2205.09575.pdf) [[Author]](https://github.com/maxwass)


1. **Self-organization Preserved Graph Structure Learning with Principle of Relevant Information.** *Qingyun Sun et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2301.00015.pdf) [[Author]](https://sunqysunqy.github.io/)

1. **Regularized Graph Structure Learning with Semantic Knowledge for Multi-variates Time-Series Forecasting.** *Hongyuan Yu et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2210.06126.pdf) [[Author]]()

1. **DBGSL: Dynamic Brain Graph Structure Learning.** *Alexander Campbell et al, .*  [[PDF]](https://arxiv.org/pdf/2209.13513.pdf) [[Author]]()

1. **Position-aware Structure Learning for Graph Topology-imbalance by Relieving Under-reaching and Over-squashing.** *Qingyun Sun et al, CIKM 2022.*  [[PDF]](https://arxiv.org/pdf/2208.08302.pdf) [[Author]](https://sunqysunqy.github.io/)


1. **Semi-Supervised Clustering via Dynamic Graph Structure Learning.** *Huaming Ling et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2209.02513.pdf) [[Author]]()


1. **Boosting Graph Structure Learning with Dummy Nodes.** *Xin Liu et al, ICML 2022.*  [[PDF]](https://arxiv.org/pdf/2206.08561.pdf) [[Author]](https://cse.hkust.edu.hk/~xliucr/)


1. **NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classification.** *Qitian Wu et al, NeurIPS 2022.*  [[PDF]](https://openreview.net/pdf?id=sMezXGG5So) [[Author]](https://qitianwu.github.io/) [[Code]](https://github.com/qitianwu/NodeFormer)





1. **Multi-view graph structure learning using subspace merging on Grassmann manifold.** *Razieh Ghiasi et al, Multimedia Tools and Applications 2022.*  [[PDF]](https://arxiv.org/pdf/2204.05258.pdf) [[Author]]()


1. **ASGNN: Graph Neural Networks with Adaptive Structure.** *Zepeng Zhang et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2210.01002.pdf) [[Author]](https://home.zepengzhang.com/)



1. **Self-organization Preserved Graph Structure Learning with Principle of Relevant Information.** *Qingyun Sun et al, AAAI 2023.*  [[PDF]](https://arxiv.org/pdf/2301.00015.pdf) [[Author]](https://sunqysunqy.github.io/) 


1. **Differentiable Graph Module (DGM) for Graph Convolutional Networks.** *Anees Kazi et al, IEEE TPAMI 2023.* [[PDF]](https://ieeexplore.ieee.org/document/9763421) [[Author]](https://campar.in.tum.de/Main/AneesKazi.html) [[Code]](https://github.com/lcosmo/DGM_pytorch)    
Notes: The current GNNs are often restricted to the transductive setting and rely on the assumption that underlying graph is known and fixed. This paper proposes **Differentiable Graph Module (DGM)** that infers the graph directly from the data. Specifically, DGM is a learnable function that predicts edge probabilities in the graph which are optimal for the downstream task. For discrete DGM, the authors construct a sparse $k$-degree graph by using the **Gumbel-Top-$k$** trick to sample edges from the probabilities. The sampling scheme, however, does not allow the gradient of the downstream classification loss function to flow through the graph prediction branch. To solve this issue, a compound loss is designed which rewards edges involved in a correct classification and penalizes edges that led to misclassification. **Latent Graph**: the graph itself is not be explicitly given.      


1. **Latent Graph Inference using Product Manifolds.** *Haitz Saez de Oc ´ ariz Borde et al, ICLR 2023.*  [[PDF]](https://arxiv.org/pdf/2211.16199.pdf) [[Author]](https://www.linkedin.com/in/haitz-s%C3%A1ez-de-oc%C3%A1riz-borde-0933a9199/details/education/)






<a name="GNNLM" />

### GNNs+LMs

1. **Efficient and effective training of language and graph neural network models** * et al, .*  [[PDF]](https://knowledge-nlp.github.io/aaai2023/papers/016-m5gnn-poster.pdf) [[Author]]()
1. **Beyond Graph Neural Networks: A New Frontier for Large Language Models** * et al, .*  [[PDF]](https://ai.plainenglish.io/beyond-graph-neural-networks-a-new-frontier-for-large-language-models-15ccb93c50d) [[Author]]()
1. **GNN-LM: Language Modeling based on Global Contexts via GNN** * et al, .*  [[PDF]](https://openreview.net/pdf?id=BS49l-B5Bql) [[Author]]()
1. **GPT4Graph: Can Large Language Models Understand Graph Structured Data? An Empirical Evaluation and Benchmarking** * et al, .*  [[PDF]](https://arxiv.org/pdf/2305.15066.pdf) [[Author]]()
1. **Disentangled Representation Learning with Large Language Models for Text-Attributed Graphs** * et al, .*  [[PDF]](https://arxiv.org/pdf/2310.18152.pdf) [[Author]]()
1. **Empower Text-Attributed Graphs Learning with Large Language Models (LLMs)** * et al, .*  [[PDF]](https://arxiv.org/pdf/2310.09872.pdf) [[Author]]()
1. **GraphGPT: Graph Instruction Tuning for Large Language Models** * et al, .*  [[PDF]](https://arxiv.org/pdf/2310.13023.pdf) [[Author]]()
1. **Can LLMs Effectively Leverage Graph Structural Information: When and Why** * et al, .*  [[PDF]](https://arxiv.org/pdf/2309.16595.pdf) [[Author]]()





<a name="GNNpruning" />

### GNN Pruning

1. **A Unified Lottery Ticket Hypothesis for Graph Neural Networks.** *Tianlong Chen et al, ICML 2021.*  [[PDF]](http://proceedings.mlr.press/v139/chen21p/chen21p.pdf) [[Author]](https://tianlong-chen.github.io/about/)  
Notes:

1. **A Study on the Ramanujan Graph Property of Winning Lottery Tickets.** *Bithika Pal et al, ICML 2022.*  [[PDF]](https://proceedings.mlr.press/v162/pal22a/pal22a.pdf) [[Author]](https://sites.google.com/view/bithikapal/home)

1. **Rethinking Graph Lottery Tickets: Graph Sparsity Matters.** *Bo Hui et al, ICLR 2023.*  [[PDF]](https://openreview.net/pdf?id=fjh7UGQgOB) [[Author]](https://bohui.herokuapp.com/)

1. **Searching Lottery Tickets in Graph Neural Networks: A Dual Perspective.** *Kun Wang et al, ICLR 2023.*  [[PDF]](https://openreview.net/pdf?id=Dvs-a3aymPe) [[Author]](http://home.ustc.edu.cn/~wk520529/#publications)
 
1. **You Can Have Better Graph Neural Networks by Not Training Weights at All: Finding Untrained GNNs Tickets.** *Tianjin Huang et al, LoG 2022.*  [[PDF]](https://arxiv.org/pdf/2211.15335.pdf) [[Author]](https://tienjinhuang.github.io/)


 
1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()








<a name="SSLGNN" />

### Self-Supervised GNN

1. **Graph Contrastive Learning with Augmentations.** *Yuning You et al, NeurIPS 2020.*  [[PDF]]() [[Author]]()

1. **When Does Self-Supervision Help Graph Convolutional Networks?** *Yuning You et al, ICML 2020.*  [[PDF]]() [[Author]]()

1. **Self-Supervised Representation Learning via Latent Graph Prediction.** *Yaochen Xie et al, ICML 2022.* [[PDF]](https://arxiv.org/pdf/2202.08333.pdf) [[Author]](https://ycremar.github.io/)   
Notes: This paper proposes **LaGraph**, a predictive SSL framework for representation learning of graph data, based on self-supervised latent graph prediction. It makes two assumptions: a. the observed feature vector of each node in an observed graph is independently generated from a certain distribution conditioned on the corresponding latent graph; b. the conditional distribution of the observed graph is centered at the latent graph. 

1. **Automated Self-Supervised Learning for Graphs.** *Wei Jin et al, ICLR 2022.*  [[PDF]](https://arxiv.org/pdf/2106.05470.pdf) [[Author]](http://cse.msu.edu/~jinwei2/)


1. **Uncovering the Structural Fairness in Graph Contrastive Learning.** *Ruijia Wang et al, NeurIPS 2022.*  [[PDF]](https://arxiv.org/pdf/2210.03011.pdf) [[Author]](https://scholar.google.com/citations?user=DpsuBrsAAAAJ&hl=zh-CN&oi=sra)






1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="GNNPreTrain" />

### GNN Pre-Training

1. **Strategies for Pre-training Graph Neural Networks.** *Weihua Hu et al, ICLR 2020.*  [[PDF]](https://arxiv.org/pdf/1905.12265.pdf) [[Author]](https://weihua916.github.io/)

1. **GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training.** *Jiezhong Qiu et al, KDD 2020.*  [[PDF]](https://arxiv.org/pdf/2006.09963.pdf) [[Author]](http://jiezhongqiu.com/)






<a name="GNNAA" />

### GNN Adversarial Attacks


1. **Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective.** *Kaidi Xu et al, IJCAI 2019.*  [[PDF]](https://arxiv.org/pdf/1906.04214.pdf) [[Author]](https://kaidixu.com/)

1. **Empowering Graph Representation Learning with Test-Time Graph Transformation.** *Wei Jin et al, ICLR 2023.*  [[PDF]](https://openreview.net/pdf?id=Lnxl5pr018) [[Author]](http://cse.msu.edu/~jinwei2/)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



<a name="GraphDomainAdaptation" />

### Graph Domain Adaptation


1. **Graph Domain Adaptation via Theory-Grounded Spectral Regularization.** *Yuning You et al, ICLR 2023.*  [[PDF]](https://openreview.net/pdf?id=OysfLgrk8mk) [[Author]](https://yyou1996.github.io/)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="GraphDataAugmentation" />

### Graph Data Augmentation

1. **Data Augmentation for Graph Neural Networks.** *Tong Zhao et al, AAAI 2021.*  [[PDF]](https://arxiv.org/pdf/2006.06830.pdf) [[Author]](https://tzhao.io/)    
Notes: 

1. **Local Augmentation for Graph Neural Networks.** *Songtao Liu et al, ICML 2022.*  [[PDF]](https://arxiv.org/pdf/2109.03856.pdf) [[Author]](https://songtaoliu0823.github.io/)   
Notes: 


1. **Graph Data Augmentation for Graph Machine Learning: A Survey.** *Tong Zhao et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2202.08871.pdf) [[Author]](https://tzhao.io/)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()










<a name="GraphGeneration" />

### Graph Generation (Graph Diffusion Model)

1. **Fast Graph Generation via Spectral Diffusion.** *Tianze Luo et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2211.08892.pdf) [[Author]](https://www.linkedin.com/in/tianze-luo-40720a82/)

1. **DiGress: Discrete Denoising diffusion for graph generation.** *Clement Vignac et al, ICLR 2023.*  [[PDF]](https://arxiv.org/pdf/2209.14734.pdf) [[Author]](https://cvignac.github.io/)

1. **Equivariant Diffusion for Molecule Generation in 3D.** *Emiel Hoogeboom et al, ICML 2022.*  [[PDF]](https://arxiv.org/pdf/2203.17003.pdf) [[Author]](https://ehoogeboom.github.io/)

1. **Bipartite Graph Diffusion Model for Human Interaction Generation.** *Baptiste Chopin et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2301.10134.pdf) [[Author]]()


1. **Diffusion Probabilistic Models for Graph-Structured Prediction.** *Hyosoon Jang et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2302.10506.pdf) [[Author]]() [[Code]](https://github.com/hsjang0/DPM-GSP)


1. **Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions.** *Emiel Hoogeboom et al, NeurIPS 2021.*  [[PDF]](https://arxiv.org/pdf/2102.05379.pdf) [[Author]](https://ehoogeboom.github.io/) [[Code]](https://github.com/ehoogeboom/multinomial_diffusion) 

1. **GMNN: Graph Markov Neural Networks.** *Meng Qu et al, ICML 2019.*  [[PDF]](http://proceedings.mlr.press/v97/qu19a/qu19a.pdf) [[Author]](https://mnqu.github.io/) [[Code]](https://github.com/DeepGraphLearning/GMNN))

1. **Diffusion Probabilistic Models for Structured Node Classification.** *Hyosoon Jang et al, ICMLW 2023.*  [[PDF]](https://openreview.net/pdf?id=DxKGgBtaR1) [[Author]]() [[Code]](https://github.com/hsjang0/DPM-SNC)

1. **Structured Denoising Diffusion Models in Discrete State-Spaces.** *Jacob Austin et al, NeurIPS 2021.*  [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2021/file/958c530554f78bcd8e97125b70e6973d-Paper.pdf) [[Author]](https://scholar.google.com/citations?user=NWXxqQ8AAAAJ&hl=zh-CN&oi=sra) [[Code]](https://github.com/google-research/google-research/tree/master/d3pm)  
Notes: While $q(x_t|x_{t-1})$ can in theory be arbitrary, efficient training of $p_{\theta}$ is possible when $q(x_t|x_{t-1})$:  
$\quad$ a. Permits efficient sampling of $x_t$ from $q(x_t|x_0)$ for an arbitrary time $t$.  
$\quad$ b. Has a tractable expression for the forward process posterior $q(x_{t-1}|x_{t}, x_0)$. 

1. **DiGress: Discrete Denoising Diffusion for Graph Generation.** *Clement Vignac et al, ICLR 2023.*  [[PDF]](https://openreview.net/pdf?id=UaAD-Nu86WX) [[Author]](https://cvignac.github.io/) [[Code]](https://github.com/cvignac/DiGress)

1. **Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling.** *Xiaohui Chen et al, ICML 2023.*  [[PDF]](https://arxiv.org/pdf/2305.04111.pdf) [[Author]](https://sites.google.com/view/xiaohuichen)  [[Code]](https://github.com/tufts-ml/graph-generation-edge)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()















#### Survey

1. **A Survey on Deep Graph Generation: Methods and Applications.** *Yanqiao Zhu et al, LoG 2022.*  [[PDF]](https://arxiv.org/pdf/2203.06714.pdf) [[Author]](https://sxkdz.github.io/)


1. **Generative Diffusion Models on Graphs: Methods and Applications.** *Wenqi Fan et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2302.02591.pdf) [[Author]](https://wenqifan03.github.io/)


1. **A Survey on Graph Diffusion Models: Generative AI in Science for Molecule, Protein and Material.** *Mengchun Zhang et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2304.01565.pdf) [[Author]]()


1. **A Systematic Survey on Deep Generative Models for Graph Generation.** *Xiaojie Guo et al, TPAMI 2023.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9920219) [[Author]](https://sites.google.com/view/xiaojie-guo-personal-site)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()


<a name="GraphCausality" />

### Causality with Graph

1. **Learning Causal Effects on Hypergraphs.** *Jing Ma et al, KDD 2022.*  [[PDF]](https://arxiv.org/pdf/2207.04049.pdf) [[Author]](https://www.cs.virginia.edu/~jm3mr/index.html)

1. **Learning Causality with Graphs.** *Jing Ma et al, AI Magazine, 2022.*  [[PDF]](https://onlinelibrary.wiley.com/doi/epdf/10.1002/aaai.12070) [[Author]](https://www.cs.virginia.edu/~jm3mr/index.html)


1. **CLEAR: Generative Counterfactual Explanations on Graphs.** *Jing Ma et al, NeurIPS 2022.*  [[PDF]](https://openreview.net/pdf?id=YR-s5leIvh) [[Author]](https://www.cs.virginia.edu/~jm3mr/index.html)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="wltest" />

### Weisfeiler-Lehman Test

1. **How Powerful are Graph Neural Networks?** *Keyulu Xu et al, ICLR 2019.*  [[PDF]](https://arxiv.org/pdf/1810.00826.pdf) [[Author]](https://people.csail.mit.edu/keyulux/)

1. **Distance Encoding: Design Provably More Powerful Neural Networks for Graph Representation Learning.** *Pan Li et al, NeurIPS 2020.* [[PDF]](https://ieeexplore.ieee.org/document/9046288) [[Author]](https://sites.google.com/view/panli-purdue/home)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="graphInfoBottle" />

### Graph Information Bottleneck 


1. **Graph Information Bottleneck.** *Tailin Wu et al, NeurIPS 2020.*  [[PDF]](https://proceedings.neurips.cc/paper/2020/file/ebc2aa04e75e3caabda543a1317160c0-Paper.pdf) [[Author]](https://tailin.org/) [[Project]](http://snap.stanford.edu/gib/#code)   
Notes: This paper proposes **Graph Information Bottleneck (GIB)**, an information-theoretic principle inherited from [Information Bottleneck](https://arxiv.org/pdf/physics/0004057.pdf) (IB), adapted for representation learning on graph-structured data. IB provides a critical principle for representation learning: *an optimal representation should contain the minimal sufficient information for the downstream task*. Based on this, GIB aims to extract information from both the graph structure and node features and further encourages the information in learned representation to be both minimal ans sufficient. The authors further propose a variational upper bound for constraining the information from the node features and graph structure, and a variational lower bound for maximizing the information in the representation to predict the target. *The i.i.d. assumption of data points is typically used to derive variational bounds and make accurate estimation of those bounds to learning IB-based models*. However, node features of graph-structured data may be correlated. **Local-Dependence** assumption: *given the data related to the neighbors within a certain number of hops of a node $v$, the data in the rest of the graph will be independent of $v$*. 


1. **Recognizing Predictive Substructures with Subgraph Information Bottleneck.** *Junchi Yu et al, TPAMI 2021.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9537601) [[Author]](https://samyu0304.github.io/)

1. **Graph Information Bottleneck for Subgraph Recognition.** *Junchi Yu et al, ICLR 2021.*  [[PDF]](https://openreview.net/pdf?id=bM4Iqfg8M2k) [[Author]](https://samyu0304.github.io/)

1. **Improving Subgraph Recognition with Variational Graph Information Bottleneck.** *Junchi Yu et al, CVPR 2022.*  [[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Improving_Subgraph_Recognition_With_Variational_Graph_Information_Bottleneck_CVPR_2022_paper.pdf) [[Author]](https://samyu0304.github.io/)

1. **Heterogeneous Graph Information Bottleneck.** *Liang Yang et al, IJCAI 2021.*  [[PDF]](https://yangliang.github.io/pdf/ijcai21.pdf) [[Author]](https://yangliang.github.io/)

1. **InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization.** *Fan-Yun Sun et al, ICLR 2020.*  [[PDF]](https://openreview.net/pdf?id=r1lfF2NYvH) [[Author]](https://sunfanyun.com/)






<a name="deeperGNN" />

### Deeper GNN

1. **Towards Deeper Graph Neural Networks.** *Meng Liu et al, KDD 2020.* [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403076) [[Author]](https://mengliu1998.github.io/)   
Notes: 




<a name="GraphOOD" />

### Graph OOD Generalization


1. **Mind the Label Shift of Augmentation-based Graph OOD Generalization.** *Junchi Yu et al, CVPR 2023.*  [[PDF]](https://arxiv.org/pdf/2303.14859.pdf) [[Author]](https://samyu0304.github.io/)




<a name="fewshotGNN" />

### Few-Shot Learning on Graph

1. **Few-Shot Learning on Graphs.** *Chuxu Zhang et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2203.09308.pdf) [[Author]](https://chuxuzhang.github.io/index.html)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()






<a name="heterogeneousGNN" />

### Heterogeneous Graph Neural Network



1. **Heterogeneous Graph Neural Network.** *Chuxu Zhang et al, KDD 2019.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3292500.3330961) [[Author]](https://chuxuzhang.github.io/index.html)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()







### Survey

1. **Geometric Deep Learning: Going beyond Euclidean Data.** *Michael Bronstein et al, IEEE SPM 2017.* [[PDF]](https://ieeexplore.ieee.org/abstract/document/7974879) [[Author]](https://www.cs.ox.ac.uk/people/michael.bronstein/) 

1. **A Comprehensive Survey on Graph Neural Networks.** *Zonghan Wu et al, IEEE TNNLS 2021.* [[PDF]](https://ieeexplore.ieee.org/document/9046288) [[Author]](https://scholar.google.com/citations?user=SzH0tgMAAAAJ&hl=en&oi=sra)

1. **Self-Supervised Learning of Graph Neural Networks: A Unified Review.** *Yaochen Xie et al, IEEE TPAMI 2023.* [[PDF]](https://arxiv.org/pdf/2102.10757.pdf) [[Author]](https://ycremar.github.io/)  

1. **A Survey on Graph Structure Learning: Progress and Opportunities.** *Yanqiao Zhu et al, arXiv 2022.* [[PDF]](https://arxiv.org/pdf/2103.03036.pdf) [[Author]](https://sxkdz.github.io/) 





<a name="VQA" />


## Visual Question Answering [[Back to Top]](#)

1. **VQA: Visual Question Answering.** *Stanislaw Antol et al, ICCV 2015.*  [[PDF]](https://openaccess.thecvf.com/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf) [[Author]](https://computing.ece.vt.edu/~santol/)


1. **Hierarchical Question-Image Co-Attention for Visual Question Answering.** *Jiasen Lu et al, NIPS 2016.*  [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2016/file/9dcb88e0137649590b755372b040afad-Paper.pdf) [[Author]](https://jiasenlu.github.io/)




1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()






<a name="ImplicitNerualRepresent" />

## Implicit Neural Representations [[Back to Top]](#)

1. **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.** *Ben Mildenhall et al, ECCV 2020.*  [[PDF]](https://arxiv.org/pdf/2003.08934.pdf) [[Author]](https://bmild.github.io/)

1. **Implicit Neural Representations with Periodic Activation Functions.** *Vincent Sitzmann et al, NeurIPS 2020.*  [[PDF]](https://proceedings.neurips.cc/paper/2020/file/53c04118df112c13a8c34b38343b9c10-Paper.pdf) [[Author]](https://www.vincentsitzmann.com/) [[Code]](https://github.com/vsitzmann/siren) [[Project]](https://www.vincentsitzmann.com/siren/)   
Notes: Most previous implicit neural representations constructed on ReLU-based MLPs lack the capacity to represent fine details in the underlying signals, and typically do not represent the derivatives of a target signal well. This is partly due to the fact that ReLU networks are piecewise linear, their second derivative is zero everywhere, and they are thus incapable of modeling information contained in higher-order derivatives of natural signals. To tackle these problems, this paper proposes **Sinusoidal Representation Networks (SIRENs)** that leverages periodic activation functions for implicit neural representations. Given a class of functions $\Phi$ that satisfy equations of the form: $$\mathcal{C}(x, \Phi, \nabla_{x} \Phi, \nabla_{x}^{2} \Phi, ...)$$in this implicit problem formulation, a functional $\mathcal{C}$ takes as input the spatio-temporal coordinates $x \in \mathbb{R}^m$ and, optional, derivatives of $\Phi$ with respect to these coordinates. The goal is to learn a neural network that parameterizes $\Phi$ to map $x$ to some quantity of interest while satisfying the constraint presented in the above equation. $\Phi$ is implicitly defined by the relation modeled by $\mathcal{C}$ and we refer to $\Phi$ as **implicit neural representations**.        



1. **Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans.** *Sida Peng et al, CVPR 2021.*  [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Peng_Neural_Body_Implicit_Neural_Representations_With_Structured_Latent_Codes_for_CVPR_2021_paper.pdf) [[Author]](https://pengsida.net/)   

1. **Learning Continuous Image Representation with Local Implicit Image Function.** *Yinbo Chen et al, CVPR 2021.*  [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.pdf) [[Author]](https://yinboc.github.io/)


1. **Implicit Neural Representations with Structured Latent Codes for Human Body Modeling.** *Sida Peng et al, TPAMI 2023.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10045794) [[Author]](https://pengsida.net/)


1. **Generalised Implicit Neural Representations.** *Daniele Grattarola et al, NeurIPS 2022.*  [[PDF]](https://openreview.net/pdf?id=2fD1Ux9InIW) [[Author]](https://danielegrattarola.github.io/)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()







<a name="deepGenerativeModels" />

## Deep Generative Models [[Back to Top]](#)


### Survey

1. **Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models.** *Sam Bond-Taylor et al, TPAMI 2022.*  [[PDF]](https://arxiv.org/pdf/2103.04922.pdf) [[Author]](https://samb-t.github.io/)




<a name="VAEs" />

### Variational Autoencoders

1. **Auto-Encoding Variational Bayes.** *Diederik P. Kingma et al, arXiv 2013.*  [[PDF]](https://arxiv.org/pdf/1312.6114.pdf) [[Author]](http://www.dpkingma.com/)

1. **An Introduction to Variational Autoencoders.** *Diederik P. Kingma et al, arXiv 2013.*  [[PDF]](https://arxiv.org/pdf/1906.02691.pdf) [[Author]](http://www.dpkingma.com/)


1. **Improved Variational Inference with Inverse Autoregressive Flows.** *Diederik P. Kingma et al, NIPS 2016.*  [[PDF]](https://arxiv.org/pdf/1606.04934.pdf) [[Author]](http://www.dpkingma.com/) [[Code]](https://github.com/openai/iaf)

1. **Ladder Variational Autoencoders.** *Casper Kaae Sønderby et al, NIPS 2016.*  [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2016/file/6ae07dcb33ec3b7c814df797cbda0f87-Paper.pdf) [[Author]](https://casperkaae.github.io/)


1. **Neural Discrete Representation Learning.** *Aaron van den Oord et al, NIPS 2017.*  [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf) [[Author]](https://avdnoord.github.io/homepage/)




1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



<a name="GANs" />

### Generative Adversarial Networks


1. **Generative Adversarial Nets.** *Ian J. Goodfellow et al, NeurIPS 2014.*  [[PDF]](https://arxiv.org/pdf/1406.2661.pdf) [[Author]](https://www.iangoodfellow.com/) [[Code]](https://github.com/goodfeli/adversarial)

1. **Conditional Generative Adversarial Nets.** *Mehdi Mirza et al, arXiv 2014.*  [[PDF]](https://arxiv.org/pdf/1411.1784.pdf) [[Author]](https://memimo.net/)


1. **Coupled Generative Adversarial Networks.** *Ming-Yu Liu et al, NeurIPS 2016.*  [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2016/file/502e4a16930e414107ee22b6198c578f-Paper.pdf) [[Author]](https://mingyuliu.net/) [[Code]](https://github.com/mingyuliutw/cogan) 



1. **Alias-Free Generative Adversarial Networks.** *Tero Karras et al, NeurIPS 2021.*  [[PDF]](https://proceedings.neurips.cc/paper/2021/file/076ccd93ad68be51f23707988e934906-Paper.pdf) [[Author]](https://research.nvidia.com/person/tero-karras)

1. **Image-to-Image Translation with Conditional Adversarial Networks.** *Phillip Isola et al, CVPR 2017.*  [[PDF]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf) [[Author]](http://web.mit.edu/phillipi/) [[Code]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) [[Website]](https://phillipi.github.io/pix2pix/)

1. **Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.** *Jun-Yan Zhu et al, ICCV 2017.*  [[PDF]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf) [[Author]]() [[Code]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) [[Website]](https://junyanz.github.io/CycleGAN/)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



<a name="normalizingFlows" />

### Normalizing Flows

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



<a name="autoregress" />

### Autoregressive Models

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



<a name="DiffusionModels" />

### Diffusion Models

1. **Deep Unsupervised Learning using Nonequilibrium Thermodynamics.** *Jascha Sohl-Dickstein et al, ICML 2015.*  [[PDF]](http://proceedings.mlr.press/v37/sohl-dickstein15.pdf) [[Author]](http://www.sohldickstein.com/)



1. **Denoising Diffusion Probabilistic Models.** *Jonathan Ho et al, NeurIPS 2020.*  [[PDF]](https://arxiv.org/pdf/2006.11239.pdf) [[Author]](http://www.jonathanho.me/) [[Code]](https://github.com/hojonathanho/diffusion)



1. **Variational Diffusion Models.** *Diederik P. Kingma et al, NeurIPS 2021.*  [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2021/file/b578f2a52a0229873fefc2a4b06377fa-Paper.pdf) [[Author]](http://www.dpkingma.com/)



1. **Denoising Diffusion Implicit Models.** *Jiaming Song et al, ICLR 2021.*  [[PDF]](https://arxiv.org/pdf/2010.02502.pdf) [[Author]](https://tsong.me/) [[Code]](https://github.com/ermongroup/ddim)

1. **Improved Denoising Diffusion Probabilistic Models.** *Alex Nichol et al, ICML 2021.*  [[PDF]](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf) [[Author]](https://aqnichol.com/) [[Code]](https://github.com/openai/improved-diffusion)


1. **Diffusion Models Beat GANs on Image Synthesis.** *Prafulla Dhariwal et al, NeurIPS 2021.*  [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf) [[Author]](https://prafulladhariwal.com/)


1. **Classifier-Free Diffusion Guidance.** *Jonathan Ho et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2207.12598.pdf) [[Author]](http://www.jonathanho.me/)

1. **Progressive Distillation for Fast Sampling of Diffusion Models.** *Tim Salimans et al, ICLR 2022.*  [[PDF]](https://arxiv.org/pdf/2202.00512.pdf) [[Author]](https://scholar.google.nl/citations?hl=en&user=w68-7AYAAAAJ&view_op=list_works&sortby=pubdate) [[Code]](https://github.com/google-research/google-research/tree/master/diffusion_distillation) 


1. **GeoDiff: a Geometric Diffusion Model for Molecular Conformation Generation.** *Minkai Xu et al, ICLR 2022.*  [[PDF]](https://arxiv.org/pdf/2203.02923.pdf) [[Author]](https://minkaixu.com/)




1. **High-Resolution Image Synthesis with Latent Diffusion Models.** *Robin Rombach et al, CVPR 2022.*  [[PDF]](https://arxiv.org/pdf/2112.10752.pdf) [[Author]](https://scholar.google.com/citations?user=ygdQhrIAAAAJ&hl=zh-CN&oi=sra) [[Code1]](https://github.com/CompVis/stable-diffusion) [[Code2]](https://github.com/CompVis/latent-diffusion)


1. **Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding.** *Chitwan Saharia et al, NeurIPS 2022.*  [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2022/file/ec795aeadae0b7d230fa35cbaf04c041-Paper-Conference.pdf) [[Author]](https://scholar.google.co.in/citations?user=JApued4AAAAJ&hl=en)


1. **Hierarchical Text-Conditional Image Generation with CLIP Latents.** *Aditya Ramesh et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2204.06125.pdf) [[Author]](http://adityaramesh.com/about.html)


1. **GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models.** *Alex Nichol et al, 2022 arXiv.*  [[PDF]](https://arxiv.org/pdf/2112.10741.pdf) [[Author]](https://aqnichol.com/)


1. **Elucidating the Design Space of Diffusion-Based Generative Models.** *Tero Karras et al, NeurIPS 2022.*  [[PDF]](https://openreview.net/pdf?id=k7FuTOWMOc7) [[Author]](https://research.nvidia.com/person/tero-karras)



1. **Understanding Diffusion Models: A Unified Perspective.** *Calvin Luo et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2208.11970.pdf) [[Author]](https://calvinyluo.com/about.html)

1. **DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps.** *Cheng Lu et al, NeurIPS 2022.*  [[PDF]](https://arxiv.org/pdf/2206.00927.pdf) [[Author]](https://luchengthu.github.io/) [[Code]](https://github.com/LuChengTHU/dpm-solver) 


1. **Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models.** *Fan Bao et al, ICLR 2022.*  [[PDF]](https://arxiv.org/pdf/2201.06503.pdf) [[Author]](https://baofff.github.io/)


1. **On Distillation of Guided Diffusion Models.** *Chenlin Meng et al, CVPR 2023.*  [[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Meng_On_Distillation_of_Guided_Diffusion_Models_CVPR_2023_paper.pdf) [[Author]](https://cs.stanford.edu/~chenlin/)



1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



#### Diffusion Pruning

1. **Structural Pruning for Diffusion Models.** *Gongfan Fang et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2305.10924.pdf) [[Author]](https://fangggf.github.io/) [[Code]](https://github.com/VainF/Diff-Pruning) 



#### Survey 

1. **Diffusion Models in Vision: A Survey.** *Florinel-Alin Croitoru et al, TPAMI 2022.*  [[PDF]](https://arxiv.org/pdf/2209.04747.pdf) [[Author]](https://scholar.google.com/citations?user=RyD1dScAAAAJ&hl=zh-CN&oi=sra)



<a name="scoreGM" />

### Score-based Generative Models



1. **Bayesian Learning via Stochastic Gradient Langevin Dynamics.** *Max Welling et al, ICML 2011.*  [[PDF]](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) [[Author]](https://scholar.google.com/citations?user=8200InoAAAAJ&hl=en)

1. **Generative Modeling by Estimating Gradients of the Data Distribution.** *Yang Song et al, NeurIPS 2019.*  [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf) [[Author]](https://yang-song.net/)

1. **Improved Techniques for Training Score-Based Generative Models.** *Yang Song et al, NeurIPS 2020.*  [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2020/file/92c3b916311a5517d9290576e3ea37ad-Paper.pdf) [[Author]](https://yang-song.net/)
2. 
1. **Score-Based Generative Modeling through Stochastic Differential Equations.** *Yang Song et al, ICLR 2021.*  [[PDF]](https://arxiv.org/pdf/2011.13456.pdf) [[Author]](https://yang-song.net/) [[Code]](https://github.com/yang-song/score_sde_pytorch)



1. **How to Train Your Energy-Based Models.** *Yang Song et al, arXiv 2021.*  [[PDF]](https://arxiv.org/pdf/2101.03288.pdf) [[Author]](https://yang-song.net/)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()














<a name="networkcompression" />

## Network Compression [[Back to Top]](#)





<a name="pruning" />

### Pruning

1. **The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.** *Jonathan Frankle et al,  ICLR 2019.*  [[PDF]](https://arxiv.org/pdf/1803.03635.pdf) [[Author]](http://www.jfrankle.com/)   
Notes: This paper proposes the **Lottery Ticket Hypothesis (LTH)**: *A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations.* Before this paper, contemporary experience is that the architectures uncovered by pruning are harder to train from the start, reaching lower accuracy than the original networks. This paper finds that when the parameters of LTH are randomly reinitialized, the wining tickets no longer match the performance of the original network, offering evidence that these smaller networks do not train effectively unless they are appropriately initialized. In another word, when randomly reinitialized, winning tickets perform far worse, meaning structure alone cannot explain a winning ticket's success. The authors identify a winning ticket by training a network and pruning its smallest-magnitude weights, and then each unpruned connections' value is then reset to its initialization from original network before it was trained. The pruning process can be *one-shot*, but *iterative pruning* shows better performance.The **Lottery Ticket Conjecture** is that *Dense, randomly-initialized networks are easier to train than the sparse networks that result from pruning because there are more possible subnetworks from which training might recover a winning ticket*. A limitation is that on deeper networks (ResNet-18 and VGG-19), iterative pruning is unable to find winning tickets unless train the networks with learning rate warmup.  




1. **Rethinking the Value of Network Pruning.** *Zhuang Liu et al, ICLR 2019.*  [[PDF]](https://arxiv.org/pdf/1810.05270.pdf) [[Author]](https://liuzhuang13.github.io/) [[Code]](https://github.com/Eric-mingjie/rethinking-network-pruning)


1. **Auto Graph Encoder-Decoder for Neural Network Pruning.** *Sixing Yu et al, ICCV 2021.*  [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Auto_Graph_Encoder-Decoder_for_Neural_Network_Pruning_ICCV_2021_paper.pdf) [[Author]](https://scholar.google.com/citations?user=DT3qqe8AAAAJ&hl=zh-CN&oi=sra) 


 
1. **The Lottery Tickets Hypothesis for Supervised and Self-supervised Pre-training in Computer Vision Models.** *Tianlong Chen et al, CVPR 2021.*  [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_The_Lottery_Tickets_Hypothesis_for_Supervised_and_Self-Supervised_Pre-Training_in_CVPR_2021_paper.pdf) [[Author]](https://tianlong-chen.github.io/about/)
 
1. **The Lottery Ticket Hypothesis for Vision Transformers.** *Xuan Shen et al, AAAI 2022.*  [[PDF]](https://arxiv.org/pdf/2211.01484.pdf) [[Author]](https://scholar.google.com/citations?user=Pvj14ZUAAAAJ&hl=en)



1. **Dual Lottery Ticket Hypothesis.** *Yue Bai et al, ICLR 2022.*  [[PDF]](https://openreview.net/pdf?id=fOsN52jn25l) [[Author]](https://yueb17.github.io/)   
Notes: This paper proposes the **Dual Lottery Ticket Hypothesis (DLTH)**: *A randomly selected subnetwork from a randomly initialized dense network can be transformed into a trainable condition, where the transformed subnetwork can be trained in isolation and achieve better at least comparable performance to LTH and other strong baselines*. LTH can be seen as finding structure according to weights, because it prune the pretrained network to find mask using weight magnitude ranking; DLTH can be seen as finding weights based on a given structure, because it transforms weights for a randomly selected sparse network. To substantiate DLTH, a **Random Sparse Network Transformation (RST)** is proposed, which adopts a regularization term to borrow learning capacity and realize information extrusion from the weights that will be masked. 



1. **Topology-Aware Network Pruning using Multi-stage Graph Embedding and Reinforcement Learning.** *Sixing Yu et al, ICML 2022.*  [[PDF]](https://proceedings.mlr.press/v162/yu22e/yu22e.pdf) [[Author]](https://scholar.google.com/citations?user=DT3qqe8AAAAJ&hl=zh-CN&oi=sra) [[Code]](https://github.com/yusx-swapp/GNN-RL-Model-Compression)




1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()















<a name="knowDistil" />

### Knowledge Distillation

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()





<a name="quantization" />

### Network Quantization

1. **Training Binary Neural Networks through Learning with Noisy Supervision.** *Kai Han et al, ICML 2020.*  [[PDF]](http://proceedings.mlr.press/v119/han20d/han20d.pdf) [[Author]](https://scholar.google.com/citations?user=vThoBVcAAAAJ&hl=zh-CN) [[Code]](https://github.com/zhaohui-yang/Binary-Neural-Networks)  


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()
2. 

<a name="LRF" />

### Low-Rank Factorization

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()


### Survey


1. **Recent Advances on Neural Network Pruning at Initialization.** *Huan Wang et al, IJCAI 2022.*  [[PDF]](https://www.ijcai.org/proceedings/2022/786) [[Author]](http://huanwang.tech/)  
Notes: This is the first survey on pruning at initialization.














<a name="datasetDistil" />

## Dataset Distillation [[Back to Top]](#)


1. **Dataset Distillation.** *Tongzhou Wang et al, arXiv 2018.*  [[PDF]](https://arxiv.org/abs/1811.10959) [[Author]](https://www.tongzhouwang.info/) [[Code]](https://github.com/SsnL/dataset-distillation) [[Website]](https://www.tongzhouwang.info/dataset_distillation/)  


1. **Generalizing Dataset Distillation via Deep Generative Prior.** *George Cazenavette et al, CVPR 2023.*  [[PDF]](https://arxiv.org/pdf/2305.01649.pdf) [[Author]](https://georgecazenavette.github.io/)









### Survey


1. **Dataset Distillation: A Comprehensive Review.** *Ruonan Yu et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2301.07014.pdf) [[Author]](https://scholar.google.com/citations?user=UHP95egAAAAJ&hl=zh-CN)  




1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()





































<a name="labelnoise" />

## Learning with Label Noise [[Back to Top]](#)



<a name="SCCnoise" />

### Statistically Consistent Classifiers



1. **Classification with Noisy Labels by Importance Reweighting.** *Tongliang Liu et al, TPAMI 2015.*  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7159100) [[Author]](https://tongliang-liu.github.io/)


1. **Are Anchor Points Really Indispensable in Label-Noise Learning?.** *Xiaobo Xia et al, NeurIPS 2019.*  [[PDF]](https://arxiv.org/pdf/1906.00189.pdf) [[Author]](https://xiaoboxia.github.io/) [[Code]](https://github.com/xiaoboxia/T-Revision)  

1. **Part-dependent Label Noise: Towards Instance-dependent Label Noise.** *Xiaobo Xia et al, NeurIPS 2020.*  [[PDF]](https://arxiv.org/pdf/2006.07836.pdf) [[Author]](https://xiaoboxia.github.io/) [[Code]](https://github.com/xiaoboxia/Part-dependent-label-noise) 



1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



<a name="SICnoise" />

### Statistically Inconsistent Classifiers



1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




















































<a name="CLR" />

## Contrastive Learning [[Back to Top]](#)

1. **A Simple Framework for Contrastive Learning of Visual Representations.** *Ting Chen et al, ICML 2020.*  [[PDF]](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf) [[Author]](https://scholar.google.com/citations?user=KoXUMbsAAAAJ&hl=en)   
Notes: 


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()


































































<a name="llv" />

## Low-Level Vision [[Back to Top]](#)


<a name="HDR" />

### High Dynamic Range Imaging

1. **Ghost-free High Dynamic Range Imaging with Context-Aware Transformer.** *Zhen Liu et al, ECCV 2022.*  [[PDF]](https://arxiv.org/abs/2208.05114) [[Author]]()   
Notes: This is the first work that introduces Transformer for HDR imaging. 


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



#### Survey 

1. **Deep Learning for HDR Imaging: State-of-the-Art and Future Trends.** *Lin Wang et al, IEEE TPAMI 2021.*  [[PDF]](https://arxiv.org/pdf/2110.10394.pdf) [[Author]](https://addisonwang2013.github.io/vlislab/linwang.html)   
 




<a name="ImageSR" />

### Image Super-Resolution

1. **Cross-Scale Internal Graph Neural Network for Image Super-Resolution.** *Shangchen Zhou et al, NeurIPS 2020.*  [[PDF]](https://proceedings.neurips.cc/paper/2020/file/23ad3e314e2a2b43b4c720507cec0723-Paper.pdf) [[Author]]()


<a name="ImageLLE" />

### Image Low-Light Enhancement

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()



















































<a name="VLP" />

## Vision Language Pretraining [[Back to Top]](#)


1. **Learning Transferable Visual Models From Natural Language Supervision.** *Alec Radford et al, ICML 2021.*  [[PDF]](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf) [[Author]](https://scholar.google.com/citations?user=dOad5HoAAAAJ&hl=en)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()














































<a name="pointcloud" />

## Point Cloud [[Back to Top]](#)

1. **Dynamic Graph CNN for Learning on Point Clouds.** *Yue Wang et al, ACM TOG 2019.*  [[PDF]](https://dl.acm.org/doi/10.1145/3326362) [[Author]](https://yuewang.xyz/)



1. **Modeling Point Clouds with Self-Attention and Gumbel Subset Sampling.** *Jiancheng Yang et al, CVPR 2019.*  [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Modeling_Point_Clouds_With_Self-Attention_and_Gumbel_Subset_Sampling_CVPR_2019_paper.pdf) [[Author]](https://jiancheng-yang.com/)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()


























<a name="cause" />

## Causal Inference [[Back to Top]](#)

1. **Advances in Variational Inference.** *Cheng Zhang et al, TPAMI 2019.*  [[PDF]](https://arxiv.org/pdf/1711.05597.pdf) [[Author]](https://cheng-zhang.org/)

1. **A Causal View on Robustness of Neural Networks.** *Cheng Zhang et al, NeurIPS 2020.*  [[PDF]](https://proceedings.neurips.cc/paper/2020/file/02ed812220b0705fabb868ddbf17ea20-Paper.pdf) [[Author]](https://cheng-zhang.org/)

1. **Relating Graph Neural Networks to Structural Causal Models.** Matej Zečević* et al, arXiv 2021.*  [[PDF]](https://qiniu.pattern.swarma.org/pdf/arxiv/2109.04173.pdf) [[Author]]()


1. **Debiasing Graph Neural Networks via Learning Disentangled Causal Substructure.** *Shaohua Fan et al, NeurIPS 2022.*  [[PDF]](https://arxiv.org/pdf/2209.14107.pdf) [[Author]](https://scholar.google.com.hk/citations?user=3LxcBjkAAAAJ&hl=zh-CN)

1. **The Causal Structure of Domain Invariant Supervised Representation Learning.** *Zihao Wang et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2208.06987v4.pdf) [[Author]](https://www.linkedin.com/in/zihao-wang-2b1645123/)


1. **Generative Causal Explanations for Graph Neural Networks.** *Wanyu Lin et al, ICML 2021.*  [[PDF]](http://proceedings.mlr.press/v139/lin21d/lin21d.pdf) [[Author]](https://wanyu-lin.github.io/)





2. 
1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()

















<a name="NLP" />

## Natural Language rocessing [[Back to Top]](#)

1. **Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing.** *Pengfei Liu et al, 2022 ACM Computing Surveys.*  [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3560815) [[Author]](http://pfliu.com/)



1. **Sparks of Artificial General Intelligence: Early experiments with GPT-4.** *S´ebastien Bubeck et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2303.12712.pdf) [[Author]](http://sbubeck.com/)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()





































<a name="AI4S" />

## AI for Sciences [[Back to Top]](#)

1. **Scientific discovery in the age of artificial intelligence.** *Hanchen Wang et al, 2023 Nature.*  [[PDF]](https://wenhao-gao.github.io/assets/publications/2023_ai4sci/paper.pdf) [[Author]](https://www.hanchenw.com/)



<a name="DTA" />

### Drug-Target Binding


1. **DeepDTA: deep drug–target binding affinity prediction.** *Hakime Öztürk et al, Bioinformatics 2018.*  [[PDF]](https://academic.oup.com/bioinformatics/article-pdf/34/17/i821/50582374/bioinformatics_34_17_i821.pdf) [[Author]](https://hkmztrk.github.io/) [[Code]](https://github.com/hkmztrk/DeepDTA)

1. **PADME: A Deep Learning-based Framework for Drug-Target Interaction Prediction
.** *Qingyuan Feng et al, arXiv 2018.*  [[PDF]](https://arxiv.org/pdf/1807.09741.pdf) [[Author]]()

1. **WideDTA: prediction of drug-target binding affinity.** *Hakime Öztürk et al, arXiv 2019.*  [[PDF]](https://arxiv.org/pdf/1902.04166.pdf) [[Author]](https://hkmztrk.github.io/)

1. **AttentionDTA: prediction of drug–target binding affinity using attention model.** *Qichang Zhao et al, BIBM 2019.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8983125) [[Author]](https://scholar.google.com/citations?user=7ixMXO0AAAAJ&hl=en)

1. **Comparison Study of Computational Prediction Tools for Drug-Target Binding Affinities.** *Maha A. Thafar et al, Frontiers in chemistry 2019.*  [[PDF]](https://www.frontiersin.org/articles/10.3389/fchem.2019.00782/full) [[Author]](https://scholar.google.com/citations?user=QgCWtUQAAAAJ&hl=zh-CN&oi=sra)

1. **DeepAffinity: interpretable deep learning of compound–protein affinity through unified recurrent and convolutional neural networks.** *Mostafa Karimi et al, Bioinformatics 2019.*  [[PDF]](https://academic.oup.com/bioinformatics/article-pdf/35/18/3329/48975803/bioinformatics_35_18_3329.pdf) [[Author]](https://mostafakarimi71.github.io/index.html)

1. **GANsDTA: Predicting Drug-Target Binding Affinity Using GANs.** *Lingling Zhao et al, Frontiers in Genetics 2020.*  [[PDF]](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2019.01243/full) [[Author]]()

1. **DeepPurpose: a deep learning library for drug–target interaction prediction.** *Kexin Huang et al, Bioinformatics 2020.*  [[PDF]](https://academic.oup.com/bioinformatics/article-pdf/36/22-23/5545/50716143/btaa1005.pdf) [[Author]](https://www.kexinhuang.com/) [[Code]](https://github.com/kexinhuang12345/DeepPurpose)

1. **GraphDTA: predicting drug–target binding affinity with graph neural networks.** *Thin Nguyen et al, Bioinformatics 2021.*  [[PDF]](https://academic.oup.com/bioinformatics/article-pdf/37/8/1140/50340643/btaa921.pdf) [[Author]](https://scholar.google.com.au/citations?user=BlTfdtMAAAAJ&hl=en) [[Code]](https://github.com/thinng/GraphDTA)

1. **Prediction of drug–target binding afnity using similarity‑based convolutional neural network.** *Jooyong Shim et al, Scientific Reports 2021.*  [[PDF]](https://www.nature.com/articles/s41598-021-83679-y.pdf) [[Author]]()

1. **Deep drug-target binding affinity prediction with multiple attention blocks.** *Yuni Zeng et al, Briefings in Bioinformatics 2021.*  [[PDF]](https://academic.oup.com/bib/article-pdf/22/5/bbab117/40260749/bbab117.pdf) [[Author]](https://scholar.google.com/citations?user=y2IUBmIAAAAJ&hl=zh-CN&oi=sra)

1. **NerLTR-DTA: drug–target binding affinity prediction based on neighbor relationship and learning to rank.** *Xiaoqing Ru et al, Bioinformatics 2022.*  [[PDF]](https://academic.oup.com/bioinformatics/article-pdf/38/7/1964/49010139/btac048.pdf) [[Author]](https://scholar.google.com.hk/citations?user=5Vc3Ub8AAAAJ&hl=zh-CN)

1. **Hierarchical graph representation learning for the prediction of
drug-target binding affinity.** *Zhaoyang Chu et al, Information Sciences 2022.*  [[PDF]](https://arxiv.org/pdf/2203.11458.pdf) [[Author]](https://scholar.google.com/citations?user=81U4yEAAAAAJ&hl=zh-CN&oi=sra)

1. **Co-VAE: Drug-Target Binding Affinity Prediction by Co-Regularized Variational Autoencoders.** *Tianjiao Li et al, TPAMI 2022.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9576631) [[Author]]()

1. **Affinity2Vec: drug-target binding affinity prediction through representation learning, graph mining, and machine learning.** *Maha A. Thafar et al, Scientific Reports 2022.*  [[PDF]](https://www.nature.com/articles/s41598-022-08787-9) [[Author]](https://scholar.google.com/citations?user=QgCWtUQAAAAJ&hl=en)

1. **DTITR: End-to-end drug–target binding affinity prediction with transformers.** *Nelson R. C. Monteiro et al, Computers in Biology and Medicine 2022.*  [[PDF]](https://www.sciencedirect.com/science/article/pii/S0010482522005406) [[Author]](https://scholar.google.com/citations?user=onq9LwwAAAAJ&hl=pt-PT)

1. **MGraphDTA: deep multiscale graph neural network for explainable drug–target binding affinity prediction.** *Ziduo Yang et al, 2022.*  [[PDF]](https://pubs.rsc.org/en/content/articlelanding/2022/sc/d1sc05180f) [[Author]](https://scholar.google.com/citations?user=w21CvkUAAAAJ&hl=zh-CN)

1. **Improving the generalizability of protein-ligand binding predictions with AI-Bind
.** *Ayan Chatterjee et al, Nature Communications 2023.*  [[PDF]](https://www.nature.com/articles/s41467-023-37572-z.pdf) [[Author]](https://chatterjeeayan.github.io/) [[Code]](https://zenodo.org/records/7730755)


1. **.** * et al, .*  [[PDF]]() [[Author]]()









<a name="others" />

## Others [[Back to Top]](#)


<a name="VI" />

### Variational Inference


1. **Variational Inference: A Review for Statisticians.** *David Blei et al, arXiv 2018.*  [[PDF]](https://arxiv.org/pdf/1601.00670.pdf) [[Author]](https://scholar.google.com/citations?user=8OYE6iEAAAAJ&hl=zh-CN&oi=sra)




<a name="procrustes" />

### Procrustes Problem

1. **A Generalized Solution of the Orthogonal Procrustes Problem.** *Peter H. Schönemann, Psychometrika 1966.*  [[PDF]](https://web.stanford.edu/class/cs273/refs/procrustes.pdf) [[Author]](https://en.wikipedia.org/wiki/Peter_Sch%C3%B6nemann)   
Notes: This is a classical paper that proposes a generalized solution to the **Orthogonal Procrustes Problem**, which is applicable to the case where the matrices involved are of less than full column rank. 

1. **<font color=blue>Generalized Embedding Regression: A Framework for Supervised Feature Extraction.</font>** *Jianglin Lu et al, IEEE TNNLS 2022.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9248650)   
Notes: My first-author paper attempts to unify previous hand-crafted feature extraction methods in a **Generalized Embedding Regression (GER)** framework. Based on GER, a new supervised feature extraction method is further proposed, which adopts the penalty graph Laplacian as the constraint matrix of a generalized orthogonal constraint. We theoretically demonstrate that the resulted optimization subproblem is intrinsically an unbalanced Procrustes problem, and elaborately design an iterative algorithm to solve it with convergence guarantee. Although the topic is somewhat out-of-date, the optimization makes me excited.




<a name="labelPropagation" />

### Label Propagation

1. **Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions** *Xiaojin Zhu, ICML 2003.*  [[PDF]](https://mlg.eng.cam.ac.uk/zoubin/papers/zgl.pdf) [[Author]](https://pages.cs.wisc.edu/~jerryzhu/)   
Notes: 

1. **Label Propagation Through Linear Neighborhoods** *Fei Wang, ICML 2006.*  [[PDF]](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=a7cd9ab6aaabf54350554f3f64c0bf19f85e65e5) [[Author]](https://wcm-wanglab.github.io/index.html)   
Notes: 



1. **Label Propagation with Weak Supervision** *, ICLR 2023.*  [[PDF]]() [[Author]]()   
Notes: 







<a name="cur" />

### CUR Decomposition

1. **Optimal CUR Matrix Decompositions.** *Christos Boutsidis et al, SIAM 2017.*  [[PDF]](https://arxiv.org/pdf/1405.7910.pdf) [[Author]](https://www.boutsidis.org/)

1. **Joint Active Learning with Feature Selection via CUR Matrix Decomposition.** *Changsheng Li et al, IEEE TPAMI 2019.*  [[PDF]](https://ieeexplore.ieee.org/document/8367893) [[Author]](https://cs.bit.edu.cn/szdw/jsml/gjjgccrc/lcs_e253eb02bdf246c4a88e1d2499212546/index.htm)   
Notes: This work performs sample selection and feature selection simultaneously based on CUR decomposition.

1. **Robust CUR Decomposition: Theory and Imaging Applications.** *HanQin Cai et al, SIAM 2021.*  [[PDF]](https://arxiv.org/pdf/2101.05231.pdf) [[Author]](https://hqcai.org/)   
Notes: This paper considers the use of Robust PCA in a CUR decomposition framework.


1. **Efficient k-NN Search with Cross-Encoders using Adaptive Multi-Round CUR Decomposition.** *Nishant Yadav et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2305.02996) [[Author]](https://nishant2yadav.github.io/)   



<a name="matrixcompletion" />

### Matrix Completion

1. **Speedup Matrix Completion with Side Information: Application to Multi-Label Learning.** *Miao Xu et al, NIPS 2013.*  [[PDF]](https://proceedings.neurips.cc/paper/2013/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf) [[Author]](https://researchers.uq.edu.au/researcher/26509)   
Notes: This paper explicitly explores the side information of data for matrix completion, with which the number of observed entries needed for a perfect recovery of matrix M can be dramatically reduced from $O(n \ln^2 n)$ to $O(\ln n)$.


1. **Matrix Completion on Graphs.** *Vassilis Kalofolias et al, arXiv 2014.*  [[PDF]](https://arxiv.org/pdf/1408.1717.pdf) [[Author]](https://scholar.google.ch/citations?user=Bz1RQ8MAAAAJ&hl=en)  

1. **Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks.** *Federico Monti et al, NeurIPS 2017.*  [[PDF]](https://proceedings.neurips.cc/paper/2017/file/2eace51d8f796d04991c831a07059758-Paper.pdf) [[Author]](https://scholar.google.com/citations?hl=en&user=NUdNFucAAAAJ)

1. **Graph Convolutional Matrix Completion.** *Rianne van den Berg et al, KDD 2018.*  [[PDF]](https://arxiv.org/pdf/1706.02263.pdf) [[Author]](https://www.microsoft.com/en-us/research/people/rvandenberg/)   
Notes: This paper considers matrix completion for recommender systems from the point of view of
link prediction on graphs.


1. **Inductive Matrix Completion Based on Graph Neural Networks.** *Muhan Zhang et al, ICLR 2020.*  [[PDF]](https://arxiv.org/pdf/1904.12058.pdf) [[Author]](https://muhanzhang.github.io/)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()












<a name="Optimization" />

### Optimization Methods

1. **Optimization Methods for Large-Scale Machine Learning.** *L ́eon Bottou et al, SIAM 2018.*  [[PDF]](https://epubs.siam.org/doi/epdf/10.1137/16M1080173) [[Author]](https://leon.bottou.org/start)   
Notes: This paper provides a review and commentary on the past, present, and future of numerical optimization algorithms in the context of machine learning applications. 

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="BiP" />

### Bilevel Programming

1. **Smooth Bilevel Programming
for Sparse Regularization.** *Clarice Poon et al, NeurIPS 2021.*  [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2021/file/0bed45bd5774ffddc95ffe500024f628-Paper.pdf) [[Author]]()

1. **Bilevel Optimization Portal.**  [[Website]](https://bi-level.org/)


<a name="PACLearning" />

### Probably Approximately Correct (PAC) Learning

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()




<a name="informationTheory" />

### Information Theory

1. **Deep Variational Information Bottleneck.** *Alexander A. Alemi et al, ICLR 2017.*  [[PDF]](https://arxiv.org/abs/1612.00410) [[Author]](https://www.alexalemi.com/)

1. **Visual Information Theory.** *Christopher Olah et al, Blog 2015.*  [[PDF]](http://colah.github.io/posts/2015-09-Visual-Information/#fnref3) [[Author]](http://colah.github.io/)

1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()





<a name="ProbabilityTheory" />

### Probability Theory

1. **Probability Theory: The Logic of Science.** *E. T. Jaynes et al, Book.*  [[PDF]](http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf) [[Author]](http://theor.jinr.ru/~kuzemsky/etjaybio.html)



<a name="GumbelmaxTrick" />

### Gumbel-Max Trick

1. **A Review of the Gumbel-max Trick and its Extensions for Discrete Stochasticity in Machine Learning.** *Iris A. M. Huijben et al, TPAMI 2023.*  [[PDF]](https://arxiv.org/pdf/2110.01515.pdf) [[Author]](https://scholar.google.com/citations?user=1ReBr6sAAAAJ&hl=en)



1. **Categorical Reparameterization with Gumbel-Softmax.** *Eric Jang et al, ICLR 2017.*  [[PDF]](https://arxiv.org/pdf/1611.01144.pdf) [[Author]](https://evjang.com/about/)


1. **Updating....** * et al, .*  [[PDF]]() [[Author]]()


<a name="SDE" />

### Stochastic Differential Equations

1. **An Intuitive Introduction For Understanding and Solving Stochastic Differential Equations.** *Chris Rackauckas et al, 2014.*  [[PDF]](https://www.ics.uci.edu/~xhx/courses/math227c/CourseScribeNotes.pdf) [[Author]](https://chrisrackauckas.com/index.html)


<a name="quantumcomputing" />

### Quantum Computing

1. **My Personal Learning Notes on Quantum Computing.** *Jianglin Lu.* [[PDF]](https://jianglin954.github.io/files/Quantum%20Computing.pdf)



























<a name="learningsources" />

## Learning Sources <a href="#top">[Back to Top]</a>

1. **UvA Deep Learning Tutorials.** [[Website]](https://uvadlc-notebooks.readthedocs.io/en/latest/index.html)   

1. **PyTorch Image Models (timm) Documentation** [[Website]](https://github.com/rwightman/pytorch-image-models)

1. **PyTorch Geometric (PyG) Documentation** [[Website]](https://pytorch-geometric.readthedocs.io/en/latest/)

1. **Deep Graph Library (DGL) Tutorials and Documentation** [[Website]](https://docs.dgl.ai/en/latest/)

1. **PyTorch Lightning Documentation** [[Website]](https://pytorch-lightning.readthedocs.io/en/stable/)

1. **KeOps Documentation** [[Website]](https://www.kernel-operations.io/keops/introduction/why_using_keops.html)

1. **Qiskit Machine Learning Documentation** [[Website]](https://qiskit.org/documentation/machine-learning/index.html)

1. **Interpretable Machine Learning** [[Website]](https://christophm.github.io/interpretable-ml-book/)






