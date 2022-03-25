<div align="center">
<h1>Azimuth Normalization </h1>

## AziNorm: Exploiting the Radial Symmetry of Point Cloud
## for Azimuth-Normalized 3D Perception

by
<br>
Shaoyu Chen, <a href="https://xinggangw.info/">Xinggang Wang</a><sup><span>&#8224;</span></sup>, Tianheng Cheng, Wenqiang Zhang, <a href="https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN">Qian Zhang</a>, <a href="https://scholar.google.com/citations?user=IyyEKyIAAAAJ&hl=zh-CN">Chang Huang</a>, <a href="http://eic.hust.edu.cn/professor/liuwenyu/"> Wenyu Liu</a>
</br>
(<span>&#8224;</span>: corresponding author)


<div>Paper: <a href="https://arxiv.org/abs/2203.13090">[arXiv version] </a><a href="">  [CVPR version]</a></div> 

</div>




## Introduction
Studying the inherent symmetry of data is of great importance in machine learning. Point cloud, the most important data format for 3D environmental perception, is naturally endowed with strong radial symmetry. In this work, we exploit this radial symmetry via a divide-and-conquer strategy to boost 3D perception performance and ease optimization. We propose Azimuth Normalization (AziNorm), which normalizes the point clouds along the radial direction and eliminates the variability brought by the difference of azimuth. AziNorm can be flexibly incorporated into most LiDAR-based perception methods. To validate its effectiveness and generalization ability, we apply AziNorm in both object detection and semantic segmentation. For detection, we integrate AziNorm into two representative detection methods, the one-stage SECOND detector and the state-of-the-art two-stage PV-RCNN detector. Experiments on Waymo Open Dataset demonstrate that AziNorm improves SECOND and PV-RCNN by 7.03 mAPH and 3.01 mAPH respectively. For segmentation, we integrate AziNorm into KPConv. On SemanticKitti dataset, AziNorm improves KPConv by 1.6/1.1 mIoU on val/test set. Besides, AziNorm remarkably improves data efficiency and accelerates convergence, reducing the requirement of data amounts or training epochs by an order of magnitude. SECOND w/ AziNorm can significantly outperform fully trained vanilla SECOND, even trained with only 10\% data or 10\% epochs.




![framework](./docs/framework.png)

<div align="center">
<img src=./docs/curve.png width=60% />
</div>
<!-- ![curve]() -->


![3D_detection](./docs/3D_detection.png)


Code and models (based on SECOND, based on Waymo and KITTI dataset) are coming soon.


## Citing AziNorm

If you find AziNorm is useful in your research or applications, please consider giving us a star &#127775; and citing AziNorm by the following BibTeX entry.

```BibTeX
@inproceedings{Chen2022AziNorm,
  title     =   {AziNorm: Exploiting the Radial Symmetry of Point Cloud for Azimuth-Normalized 3D Perception},
  author    =   {Shaoyu Chen, Xinggang Wang, Tianheng Cheng, Wenqiang Zhang, Qian Zhang, Chang Huang and Wenyu, Liu},
  booktitle =   {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =   {2022}
}

```
