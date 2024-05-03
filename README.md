# YOLOv9

Implementation of paper - [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)

## Segmentation Predict only

[`gelan-c-seg.pt`](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c-seg.pt)

| Model | Test Size | Param. | FLOPs | AP<sup>box</sup> | AP<sup>mask</sup>  |
| :-- | :-: | :-: | :-: | :-: | :-: |
| [**GELAN-C-SEG**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c-seg.pt) | 640 | 27.4M | 144.6G | **52.3%** | **42.4%** |

## Installation & Prediction

<details><summary><b>Conda Environment Setup</b></summary>

``` shell
# Create the Conda environment
conda env create -f environment.yml

# Activate the environment
conda activate yolov9segconda

# Run prediction
python predict.py

```

</details>





## Citation

```
@article{wang2024yolov9,
  title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
  author={Wang, Chien-Yao  and Liao, Hong-Yuan Mark},
  booktitle={arXiv preprint arXiv:2402.13616},
  year={2024}
}
```

```
@article{chang2023yolor,
  title={{YOLOR}-Based Multi-Task Learning},
  author={Chang, Hung-Shuo and Wang, Chien-Yao and Wang, Richard Robert and Chou, Gene and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2309.16921},
  year={2023}
}
```



## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/VDIGPKU/DynamicDet](https://github.com/VDIGPKU/DynamicDet)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)

</details>
