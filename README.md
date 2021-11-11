# Reinforcement Explanation Learning (RExL)

## Setup Instructions

1. Install the RISE package (to use Causal Metrics)
```
cd RISE
pip install -e .
```

2. Install the explainableAI env
```
cd RExL
pip install -e .
```

3. Note the location of the dataset and the base models. Update the paths for the base models accordingly in ```RExL/explainableAI/utils/models.py```. The pretrained models are taken from [here](https://github.com/jimmie33/Caffe-ExcitationBP/blob/master/excitationBP/Demo_resnet.ipynb). These models are in caffe so these must be converted to pytorch.

## Training Instructions

Run ```RExL/train.py``` with the following arguments:

```-rp``` or ```--root_path``` : Root path for the Dataset

```-m``` or ```--model``` : Model type ```resnet``` or ```vgg```

```-ci``` or ```--class_index``` : Class index for the class on which the agent is to be trained (```-1``` if training Dataset Specific)

```-d``` or ```--dataset``` : Dataset Name eg: ```PASCAL```, ```MSCOCO``` or ```IMAGENET```

```-dt``` or ```--dataset_type```: Dataset type, eg: ```train``` or ```val``` or ```test```

```-tl``` or ```--tensorboard_log_dir```: Path for tensorboard logs

```-sp``` or ```--save_path```: Path to save the trained policy

```-nt``` or ```--num_timesteps```: Number of training steps

```-si``` or ```--save_interval```: Interval to periodically save the policy

```-lp``` or ```--load_path```: Path to load a previously trained policy, Default: ```None``` i.e not applicable

```-vp``` or ```--video_path```: Path to store the images for each step. Default: ```None``` i.e. not applicable

```-i``` or ```--idx```: Id of a specific image to be trained on (RExL-IS)

## Evaluating Instructions

Run ```RExL/evaluate.py``` with the following arguments:

```-rp``` or ```--root_path``` : Root path for the Dataset

```-m``` or ```--model``` : Model type ```resnet``` or ```vgg```

```-ci``` or ```--class_index``` : Class index for the class on which the agent is to be trained (```-1``` if training Dataset Specific)

```-d``` or ```--dataset``` : Dataset Name eg: ```PASCAL```, ```MSCOCO``` or ```IMAGENET```

```-dt``` or ```--dataset_type```: Dataset type, eg: ```train``` or ```val``` or ```test```

```-lp``` or ```--load_path```: Path to load a previously trained policy, Default: ```None``` i.e not applicable

```-bs``` or ```--batch_size```: Batch size for running causal metrics

```-v``` or ```--verbose```: ```0``` (Default) if saliency maps are not saved and ```1``` to save saliency maps. ```-v=1``` works with ```-bs=1``` only.

```-ip``` or ```--image_path```: Path to save the images

```-log``` or ```--log_path```: Path to save the image wise logs

```-i``` or ```--idx```: Id of a specific image to be trained on (RExL-IS)