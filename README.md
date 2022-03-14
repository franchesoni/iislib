
# Interactive Image Segmentation Framework

This is a library that exposes simple but powerful modules to do research on Interactive Image Segmentation (IIS) using deep learning (DL).


In an IIS system we have many components, organized as the folders of the project or implementing one of the decisions below.

## High level overview
This framework is aimed at researchers on IIS. An IIS step can be summarized in:
1. grab an image and (optionally) a segmentation candidate
2. oracle annotates the image (e.g. robot click)
3. image and annotation are given to model, which produces a new segmentation candidate

This can be further divided in smaller blocks. If we assume that the annotation is a robot click, we have that we can evaluate a model over a segmentation database as in the following **pseudocode**:

```python
for img, gt_mask in segmentation_database:
    target_region = sample_region(gt_mask)
    prev_output = zeros_like(gt_mask)
    for interaction_ind in range(interaction_steps):
        clicks = robot_click(prev_output, target_region)
        aux_input = encode_aux(clicks, prev_output)
        prev_output = model(img, aux_input)
    save_metrics(prev_output, target_region)
```
different choices for `sample_region` / `robot_click` / `encode_aux` / `model` will provide different results.

## Run
Each script has a `test` function which is usually called when running the script directly. For instance, you can run `python -m data.clicking` and see what happens :)


## Structure
- `data/`
    - `iis_dataset.py`: Abstract `SegDataset` (to load whatever segmentation dataset you have) and `RegionDataset` classes are here defined. Use them to retrieve image and masks or image and target region, respectively.
    - `datasets/`
        - `dataset_name.py`: each specific dataset is usually an image segmentation dataset and has an specific script to load each image and its corresponding masks. Subclass `SegDataset`
    - `region_selector.py`: given a dataset script, which retrieves a sample image and its masks, we must select one ground truth mask from between these masks. Different strategies are implemented here.
    - `clicking.py`: given a ground truth mask, past clicks and a prediction, we compute the next click according to different strategies. The prediction can be empty if it is the first step. This script also implements clicks encoding.
- `models/`: for us, IIS models are segmentation models with more input channels
    - `iis_segmenter_wrapper.py`: this is a model that is built upon a segmentation backbone (that maps an RGB image to a mask prediction) and allows for IIS inputs
    - `iis_representer_wrapper.py`: this is a model that is built upon a feature extractor or representer backbone (that maps an RGB image to a feature vector) and allows for IIS inputs
    - `backbones/`
        - `representer/`
            - `MAE_ViT.py`
        - `segmenter/`
            - `HRNet.py`
    - `iis_models/`
        - `ritm.py`
- `engine/`: training loops, metrics, and engineering
    - `metrics.py`: metrics can be computed from ground truth mask vs prediction but also counting the number of clicks is relevant
    - `training_logic.py`: here goes the various training loops
- `try_model.py`: script to train and evaluate a model using components from above

I will (try to) use a functional programming approach. This is, most things will be implemented as functions with explicit inputs and outputs (composition over inheritance). However, DL models are naturally better expressed on object-oriented paradigm.

## Decisions
The organization gives room for many decisions to be made. For instance: 
- which dataset to use? e.g. new dataset vs recommended COCO_LVIS
- how to sample ground truth regions from the dataset? e.g. randomly, ignoring background, merging
- how to sample clicks? (initial and subsequent) e.g. randomly, over error regions, over borders, depending on distance to the border
- how to encode clicks? e.g. disks, distance maps
- how to wrap a segmentation model? e.g. early vs late fusion
- how to wrap a feature extractor? e.g. which upsampler to use
- which metrics to compute? e.g. mIoU vs n_clicks
- how to do the training? e.g. iterative

## Future optimizations

- use fast jpeg reader instead of opencv
- profile and reduce training time of full example

## Datasets
One should note that datasets are different and it is not straightforward to evaluate them in general. It is easier for those who have a binary mask.


- data
    - transforms
    - RegionDataset (for training)
        - segmentation dataset 
        - region selector (random)
    - EvaluationDataset (for testing)
        - segmentation dataset 
        - region selector (trivial, as we use only some segmentation datasets)

- Clicking
    - utilities: sample uniformly, get largest region, etc.
    - robots
        - input: prediction, real mask, past clicks
        - output: click(s)
        - training types: random, random false, random target false, center 99
        - evaluation types: random, random false, random largest false, center 99, complicated RITM
    - encodings
        - disk encoding 

- Engine 
    - training logic
        - training steps: interact some steps, learn how to best correct
    - metrics
        - those in openMM

- models
    - lightning.py
    - wrappers
        - openmm
        - smp
    - custom models
        - ritm
        - 99
    
- tests
- train model
- evaluate model

## tasks
- evaluate a pretrained or not pretrained model over one clicking scheme
- evaluate 99
- evaluate ritm
- implement other clicking schemes
added:
- move get_positive_clicks_batch into a robot
- move smp wrapper inside wrappers

done
- reorganize the code a little