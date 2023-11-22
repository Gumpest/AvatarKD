# Avatar Knowledge Distillation: Self-ensemble Teacher Paradigm with Uncertainty (AvatarKD)

:fire: Official implementation of paper "Avatar Knowledge Distillation: Self-ensemble Teacher Paradigm with Uncertainty" (AvatarKD), ACM MM 2023.

By Yuan Zhang, Weihua Chen, Yichen Lu, Tao Huang, Xiuyu Sun and Jian Cao.

<p align='center'>
<img src='./assests/arch.png' alt='mask' width='700px'>
</p>

## Installation  

### Install MMRazor 0.x

```shell
git clone -b 0.x https://github.com/open-mmlab/mmrazor.git
cd mmrazor
```

```shell
pip install -v -e .
```

### Prepare Data Set

Download on [https://opendatalab.com](https://opendatalab.com)

> [!Note]
> If you want to distill on detection and segmentation, you should install mmdetection and mmsegmentation, respectively.

## Reproducing our results

### Train students with Avatars

This repo uses [MMRazor](https://github.com/open-mmlab/mmrazor) as the knowledge distillation toolkit. For environment setup, please see [docs/en/get_started.md](docs/en/get_started.md).

**Train student:**  

```shell
cd mmrazor
sh tools/mmdet/dist_train_mmdet.sh ${CONFIG} 8 ${WORK_DIR}
```

Example for reproducing our `reppoints_x101-reppoints-r50_coco` result:

```shell
sh tools/mmdet/dist_train_mmdet.sh akd_cwd_reppoints_x101-reppoints-r50_coco.py 8 work_dirs/akd_rep_x101-fpn_x50
```


### Results  

* Baseline settings:  

  |        Student         |         Teacher         | AvatarKD |
  | :--------------------: | :---------------------: | :------: |
  | Faster RCNN-R50 (38.4) | Faster RCNN-R101 (39.8) |   40.9   |
  |  RetinaNet-R50 (37.4)  |  RetinaNet-R101 (38.9)  |   40.3   |
  |    FCOS-R50 (38.5)     |    FCOS-R101 (40.8)     |   42.9   |

* Stronger teachers:

  |        Student         |            Teacher            | AvatarKD |
  | :--------------------: | :---------------------------: | :------: |
  | Faster RCNN-R50 (38.4) | Cascade Mask RCNN-X101 (45.6) |   42.4   |
  |  RetinaNet-R50 (37.4)  |     RetinaNet-X101 (41.0)     |   41.5   |
  |  RepPoints-R50 (38.6)  |     RepPoints-R101 (44.2)     |   42.8   |

### Visualization

<p align='center'>
<img src='./assests/vis.png' alt='mask' width='400px'>
</p>

## License  

This project is released under the [Apache 2.0 license](LICENSE).

## Citation  

```
@article{zhang2023avatar,
  title={Avatar Knowledge Distillation: Self-ensemble Teacher Paradigm with Uncertainty},
  author={Zhang, Yuan and Chen, Weihua and Lu, Yichen and Huang, Tao and Sun, Xiuyu and Cao, Jian},
  journal={arXiv preprint arXiv:2305.02722},
  year={2023}
}
```
