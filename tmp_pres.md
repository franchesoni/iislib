
# IIS

**objective** annotate datasets faster

for that:
- develop annotation tools (IIS)
- better train models (active learning, etc)

**problem:** generalization to new data (if we had a model for unlabeled data, we could use it to annotate)

So far:
- lit review: click-based deep learning methods (gto99, ritm)
- iislib: allows for experimentation (to be shown next)
- 2 mvasat projects
- igarss

In progress:
- correct igarss
- IPOL 1: clicking procedure comparison
- IPOL 2: mvasat1, generalization to aerial images

Future:
- mvasat2, continual adaptation to new image sequences
- attempt to mathematically formalize the problem
- optimal clicking procedure + active learning
- use robust models (e.g. SSL trained MAE). Ideas:
    - train with more similar data (mvasat1)
    - train more general models (SSL)
    - efficient pretraining (use classic segmentation as labels, then use learned segmentations as labels)
    - mix eff pretraining with clicking, using different segmentation methods
