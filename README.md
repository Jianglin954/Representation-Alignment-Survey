# Representation Potentials of Foundation Models for Multimodal Alignment: A Survey (EMNLP25)


[![Awesome Representation Alignment](https://awesome.re/badge.svg)](https://awesome.re)
![Paper Reading](https://img.shields.io/badge/Multimodal_Alignment-blue)

This is the repository for the survery paper: [Representation Potentials of Foundation Models for Multimodal Alignment: A Survey](). The collection will be continuously updated, so star (🌟) & stay tuned. Any suggestions and comments are welcome (jianglinlu@outlook.com). 



## Contents
- [Foundation Models](#FM)
  - [Vision Foundation Models](#VFMs)
  - [Large Language Models](#LLMs)
  - [Speech Foundation Models](#SFMs)
  - [Multimodal Foundation Models](#MFMs)
- [Alignment Metrics](#ALME)
- [Representation Convergence within Single-Modality](#RCSM)
  - [Vision](#RCSMV)
  - [Language](#RCSML)
  - [Speech](#RCSMS)
- [Representation Convergence within Across-Modalities](#RCAM)
- [Representation Convergence in Biological Systems](#RCBS)
- [Open Questions](#OpenQ)



<a name="FM" />

## Foundation Models [[Back to Top]](#)

1. **On the Opportunities and Risks of Foundation Models** *Rishi Bommasani et al, arXiv 2022.* [[PDF]](https://arxiv.org/pdf/2108.07258)




<a name="VFMs" />

### Vision Foundation Models

1. **Deep Residual Learning for Image Recognition** * Kaiming He et al, CVPR 2016.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7780459) 

1. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** *Alexey Dosovitskiy et al, ICLR 2021.*  [[PDF]](https://arxiv.org/pdf/2010.11929/1000)

1. **A ConvNet for the 2020s** *Zhuang Liu et al, CVPR 2022.*  [[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf) 

1. **ConvNeXt V2: Co-Designing and Scaling ConvNets With Masked Autoencoders** *Sanghyun Woo et al, CVPR 2023.*  [[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf) 

1. **DINOv2: Learning Robust Visual Features without Supervision** *Maxime Oquab et al, arXiv 2024.*  [[PDF]](https://arxiv.org/pdf/2304.07193) 

1. **DINOv3.** *Oriane Siméoni et al, arXiv 2025.*  [[PDF]](https://arxiv.org/pdf/2508.10104) 

1. **Segment Anything** *Alexander Kirillov et al, ICCV 2023.*  [[PDF]](https://arxiv.org/pdf/2304.02643) 




<a name="LLMs" />

### Large Language Models

1. **Language Models are Few-Shot Learners** *Tom B. Brown et al, NeurIPS 2020.*  [[PDF]](https://arxiv.org/pdf/2005.14165)

1. **Scaling Laws for Neural Language Models** *Jared Kaplan et al, arXiv 2020.*  [[PDF]](https://arxiv.org/pdf/2001.08361)

1. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** *Jacob Devlin et al, NAACL 2019.*  [[PDF]](https://aclanthology.org/N19-1423.pdf) 

1. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** *Colin Raffel et al, JMLR 2020.*  [[PDF]](https://arxiv.org/pdf/1910.10683)

1. **Emergent Abilities of Large Language Models** *Jason Wei et al, TMLR 2022.*  [[PDF]](https://openreview.net/pdf?id=yzkSU5zdwD) 

1. **Qwen Technical Report** *Jinze Bai et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2309.16609)

1. **The Llama 3 Herd of Models** *Aaron Grattafiori et al, arXiv 2024.*  [[PDF]](https://arxiv.org/pdf/2407.21783) 




<a name="SFMs" />

### Speech Foundation Models

1. **wav2vec: Unsupervised Pre-training for Speech Recognition** *Steffen Schneider et al, arXiv 2019.*  [[PDF]](https://arxiv.org/pdf/1904.05862)

1. **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations** *Alexei Baevski et al, NeurIPS 2020.*  [[PDF]](https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf)

1. **HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units** *Wei-Ning Hsu et al, arXiv 2021.*  [[PDF]](https://arxiv.org/pdf/2106.07447)

1. **WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing** *Sanyuan Chen et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2110.13900)

1. **Robust Speech Recognition via Large-Scale Weak Supervision** *Alec Radford et al, ICML 2023.*  [[PDF]](https://proceedings.mlr.press/v202/radford23a/radford23a.pdf)

1. **SeamlessM4T: Massively Multilingual & Multimodal Machine Translation** *Seamless Communication et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2308.11596) 




<a name="MFMs" />

### Multimodal Foundation Models

1. **Learning Transferable Visual Models From Natural Language Supervision** *Alec Radford, et al, ICML 2021.*  [[PDF]](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf) 

1. **Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision** *Chao Jia et al, ICML 2021.*  [[PDF]](https://arxiv.org/pdf/2102.05918) 

1. **Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation.** *Junnan Li et al, ICML 2022.*  [[PDF]](https://arxiv.org/pdf/2201.12086) 

1. **Coca: Contrastive captioners are image-text foundation models.** *Jiahui Yu et al, arXiv 2022.*  [[PDF]](https://arxiv.org/pdf/2205.01917) 

1. **Flamingo: a visual language model for few-shot learning.** *Jean-Baptiste Alayrac et al, NeurIPS 2022.*  [[PDF]](https://arxiv.org/pdf/2204.14198) 

1. **PaLI: A Jointly-Scaled Multilingual Language-Image Model.** *Xi Chen et al, ICLR 2023.*  [[PDF]](https://arxiv.org/pdf/2209.06794) 

1. **GPT-4 Technical Report.** *OpenAI, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2303.08774) 

1. **Gemini: A Family of Highly Capable Multimodal Models.** *Gemini Team, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2312.11805) 

1. **A Survey on Multimodal Large Language Models.** *Shukang Yin et al, National Science Review 2024.*  [[PDF]](https://arxiv.org/pdf/2306.13549) 



<a name="ALME" />

## Alignment Metrics [[Back to Top]](#)


1. **Supervised Feature Selection via Dependence Estimation.** *Le Song et al, ICML 2007.*  [[PDF]](https://arxiv.org/pdf/0704.2668) 

1. **SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability.** *Maithra Raghu et al, NIPS 2017.*  [[PDF]](https://arxiv.org/pdf/1706.05806) 

1. **Representational models: A common framework for understanding encoding, pattern-component, and representational-similarity analysis.** *Jörn Diedrichsen et al, PLoS computational biology 2017.*  [[PDF]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005508) 

1. **Insights on Representational Similarity in Neural Networks with Canonical Correlation.** *Ari S. Morcos et al, NeurIPS 2018.*  [[PDF]](https://dl.acm.org/doi/pdf/10.5555/3327345.3327475) 

1. **Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors.** *Laleh Haghverdi et al, Nature Biotechnology 2018.*  [[PDF]](https://www.nature.com/articles/nbt.4091.pdf) 

1. **Similarity of Neural Network Representations Revisited** *Simon Kornblith. et al, ICML 2019.*  [[PDF]](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf) 

1. **On the Cross-lingual Transferability of Monolingual Representations.** *Mikel Artetxe et al, ACL 2020.*  [[PDF]](https://arxiv.org/pdf/1910.11856) 

1. **Towards Understanding the Instability of Network Embedding.** *Chenxu Wang et al, TKDE 2020.*  [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9076342) 

1. **Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth.** *Thao Nguyen et al, ICLR 2021.*  [[PDF]](https://openreview.net/pdf?id=KJNcAkY8tY4) 

1. **Using distance on the Riemannian manifold to compare representations in brain and in models.** *Mahdiyar Shahbazi et al, NeuroImage 2021.*  [[PDF]](https://www.sciencedirect.com/science/article/pii/S1053811921005474) 

1. **Reliability of CKA as a Similarity Measure in Deep Learning.** *MohammadReza Davari et al, ICLR 2023.*  [[PDF]](https://openreview.net/pdf?id=8HRvyxc606) 

1. **Similarity of Neural Network Models: A Survey of Functional and Representational Measures.** *Max Klabunde et al, arXiv 2025.*  [[PDF]](https://arxiv.org/pdf/2305.06329) 



<a name="RCSM" />

## Representation Convergence within Single-Modality [[Back to Top]](#)


<a name="RCSMV" />

### Vision

1. **Understanding Image Representations by Measuring Their Equivariance and Equivalence.** *Karel Lenc et al, CVPR 2015.*  [[PDF]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Lenc_Understanding_Image_Representations_2015_CVPR_paper.pdf) 

1. **Convergent Learning: Do different neural networks learn the same representations?** *Yixuan Li et al, ICLR 2016.*  [[PDF]](https://arxiv.org/pdf/1511.07543) 

1. **SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability.** *Maithra Raghu et al, NIPS 2017.*  [[PDF]](https://arxiv.org/pdf/1706.05806) 

1. **A Spline Theory of Deep Learning.** *Randall Balestriero et al, ICML 2018.*  [[PDF]](https://proceedings.mlr.press/v80/balestriero18b/balestriero18b.pdf) 


1. **Insights on Representational Similarity in Neural Networks with Canonical Correlation.** *Ari S. Morcos et al, NeurIPS 2018.*  [[PDF]](https://dl.acm.org/doi/pdf/10.5555/3327345.3327475) 

1. **Similarity of Neural Network Representations Revisited** *Simon Kornblith. et al, ICML 2019.*  [[PDF]](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf) 

1. **Similarity and Matching of Neural Network Representations.** *Adrián Csiszárik et al, NeurIPS 2021.*  [[PDF]](https://openreview.net/pdf?id=aedFIIRRfXr) 

1. **On Linear Identifiability of Learned Representations.** *Geoffrey Roeder et al, ICML 2021.*  [[PDF]](https://proceedings.mlr.press/v139/roeder21a/roeder21a.pdf) 

1. **Do Self-Supervised and Supervised Methods Learn Similar Visual Representations?** *Tom George Grigg et al, arXiv 2021.*  [[PDF]](https://arxiv.org/pdf/2110.00528) 

1. **Revisiting Model Stitching to Compare Neural Representations.** *Yamini Bansal et al, NeurIPS 2021.*  [[PDF]](https://arxiv.org/pdf/2106.07682) 

1. **Emerging Properties in Self-Supervised Vision Transformers.** *Mathilde Caron et al, ICCV 2021.*  [[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf) 

1. **Do Vision Transformers See Like Convolutional Neural Networks?** *Maithra Raghu et al, NeurIPS 2021.*  [[PDF]](https://arxiv.org/pdf/2108.08810) 

1. **Relative Representations Enable Zero-Shot Latent Space Communication.** *Luca Moschella et al, ICLR 2023.*  [[PDF]](https://arxiv.org/pdf/2209.15430) 

1. **Objectives Matter: Understanding the Impact of Self-Supervised Objectives on Vision Transformer Representations.** *Shashank Shekhar et al, arXiv 2023.*  [[PDF]](https://arxiv.org/pdf/2304.13089) 

1. **Rosetta Neurons: Mining the Common Units in a Model Zoo.** *Amil Dravid et al, ICCV 2023.*  [[PDF]](https://arxiv.org/pdf/2306.09346) 

1. **DINOv2: Learning Robust Visual Features without Supervision.** *Maxime Oquab et al, TMLR 2024.*  [[PDF]](https://arxiv.org/pdf/2304.07193) 

1. **ZipIt! Merging Models from Different Tasks without Training.** *George Stoica et al, ICLR 2024.*  [[PDF]](https://arxiv.org/pdf/2305.03053) 

1. **The Platonic Representation Hypothesis.** *Minyoung Huh et al, ICML 2024.*  [[PDF]](https://arxiv.org/pdf/2405.07987) 

1. **How Do the Architecture and Optimizer Affect Representation Learning? On the Training Dynamics of Representations in Deep Neural Networks.** *Yuval Sharon et al, arXiv 2025.*  [[PDF]](https://arxiv.org/pdf/2405.17377) 

1. **Dual Diffusion for Unified Image Generation and Understanding.** *Zijie Li et al, CVPR 2025.*  [[PDF]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Dual_Diffusion_for_Unified_Image_Generation_and_Understanding_CVPR_2025_paper.pdf) 

1. **Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think.** *Sihyun Yu et al, ICLR 2025.*  [[PDF]](https://arxiv.org/pdf/2410.06940) 




<a name="RCSML" />

### Language

1. **Updating.** * et al, .*  [[PDF]]() 

1. **Updating.** * et al, .*  [[PDF]]() 

1. **Updating.** * et al, .*  [[PDF]]() 

1. **Updating.** * et al, .*  [[PDF]]() 

1. **Updating.** * et al, .*  [[PDF]]() 

1. **Updating.** * et al, .*  [[PDF]]() 


<a name="RCSMS" />

### Speech

1. **Updating.** * et al, .*  [[PDF]]() 

1. **Updating.** * et al, .*  [[PDF]]() 

1. **Updating.** * et al, .*  [[PDF]]() 


<a name="RCAM" />

## Representation Convergence within Across-Modalities [[Back to Top]](#)

1. **Updating.** * et al, .*  [[PDF]]() 

1. **Updating.** * et al, .*  [[PDF]]() 

1. **Updating.** * et al, .*  [[PDF]]() 


<a name="RCBS" />

## Representation Convergence in Biological Systems [[Back to Top]](#)

1. **Updating.** * et al, .*  [[PDF]]() 

1. **Updating.** * et al, .*  [[PDF]]() 

1. **Updating.** * et al, .*  [[PDF]]() 


<a name="OpenQ" />

## Open Questions [[Back to Top]](#)


1. **Updating....** * et al, .*  [[PDF]]() 







## 📝 Citation

If you find our survey useful, please consider citing:
``` bib file
@article{lu2025representations,
  title={Representation Potentials of Foundation Models for Multimodal Alignment: A Survey},
  author={Lu, Jianglin and Wang, Hailing and Xu, Yi and Wang, Yizhou and Yang, Kuo and Fu, Yun},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, {EMNLP} 2025},
  publisher={Association for Computational Linguistics},
  year={2025}
}
```

