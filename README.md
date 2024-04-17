# This is the official pytorch implementation of our ICLR24 "Debiased Collaborative Filtering with Kernel-Based Causal Balancing" paper.

We use three public real-world datasets (Coat, Music and Product) for real-world experiments.
## Run the code

- For Coat, please run the file:


```python
coat.ipynb
```


- For Music, please run the file:

```python
music.ipynb
```


- For Product, please run the file:


```python
product.ipynb 
```


The code runs well at python 3.8.18. The required packages are as follows:
-   pytorch == 1.9.0
-   numpy == 1.24.4 
-   scipy == 1.10.1
-   pandas == 2.0.3
-   scikit-learn == 1.3.2


## 
If you find this code useful for your work, please consider to cite our work as
```
@inproceedings{li2024kernel,
  title={Debiased Collaborative Filtering with Kernel-Based Causal Balancing},
  author={Li, Haoxuan and Xiao, Yanghao and Zheng, Chunyuan and Wu, Peng and Chen, Xu and Geng, Zhi and Cui, Peng},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2024}
}
```
