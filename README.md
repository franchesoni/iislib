
# Interactive Image Segmentation Framework

This is a library that exposes simple but powerful modules to do research on Interactive Image Segmentation (IIS) using deep learning (DL).


In an IIS system we have many components, organized as the folders of the project or implementing one of the decisions below.

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