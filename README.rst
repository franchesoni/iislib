Interactive Image Segmentation Framework
========================================

This is a library that exposes simple but powerful modules to do
research on Interactive Image Segmentation (IIS) using deep learning
(DL).

In an IIS system we have many components, organized as the folders of
the project. Here I’ll explain the high level interrelations between the
components. For more details, read the code.

IIS in a nutshell
-----------------

This framework is aimed at researchers on IIS. An IIS step can be
summarized in: 1. grab an image and the corresponding segmentation mask
2. oracle annotates the image given a target mask (e.g. robot click) 3.
image and annotation are given to model, which produces a new
segmentation candidate

In practice an annotation is a set of positive/negative clicks and there
are auxiliary variables that are used and transformed by the model. An
IIS model :math:`f` is then

.. math:: \hat{y}_{k+1}, z_{k+1} = f(x, z_k, [c_1, \dots, c_k])

where :math:`x` is the input image, :math:`z` is an auxiliary variable
(e.g. :math:`z_k = \hat{y}_k`), and :math:`c_k` are the clicks made at
interaction step :math:`k`.

The *pseudocode* for a full inference is the following

.. code:: python

   def full_inference(model, oracle, image, target, K):
       # initialization
       click_seq = []
       z = model.initialize_z(image, target)
       y = model.initialize_y(image, target)

       # inference
       for k in [1, ..., K]:
           clicks_k = oracle(y, target, click_seq)  # annotate
           click_seq.append(clicks_k)
           y, z = model(image, z, click_seq)  # predict
       return y

Structure
---------

-  ``data/``

   -  ``iis_dataset.py``: Classes here defined:

      -  Abstract ``SegDataset`` (subclass to load whatever segmentation
         dataset you have). Loads image, full mask and info.
      -  ``RegionDatasetWithInfo`` loads image, target region and info.
         There for compatibility, use ``RegionDataset`` instead.
      -  ``RegionDataset`` loads image and target region.
      -  ``EvaluationDataset`` is a ``RegionDataset`` but it ensures
         reproducibility (by asking for databases with binary masks).

   -  ``datasets/``

      -  ``dataset_name.py``: each specific dataset is usually an image
         segmentation dataset and has an specific script to load each
         image and its corresponding masks. Subclasses ``SegDataset``.

   -  ``region_selector.py``: given a dataset script, which retrieves a
      sample image and its masks, we must select one target mask from
      between the original masks. Different strategies are implemented
      here.
   -  ``transforms.py``: augmentations and transformations of images or
      masks

-  ``clicking/`` outputs, # (B, C, H, W) targets, n_points=1, pcs=[], #
   indexed by (interaction, batch_element, click) ncs=[],

   -  ``robots.py``: Implements clicking robots. Given a ground truth
      mask, past clicks and a prediction, we compute the next clicks
      according to different strategies. The prediction can be empty if
      it is the first step. Examples: random, random false, random
      largest false, center largest false,
   -  ``encode.py`` This script implements clicks encoding,
      i.e. transforming (i, j) coordinates into a click map (H, W).
   -  ``utils.py`` All sort of functions dealing with masks.

-  ``models/``: for us, IIS models are segmentation models with extra
   inputs or custom models from external repos.

   -  ``wrappers/``: there are powerful segmentation models we can use
      with some modification. Here we implement wrappers around modules.

      -  ``iis_smp_wrapper.py``: Implements early fusion and
         intermediate fusion for ``segmentation_models_pytorch`` module.
      -  ``iis_openmm_wrapper.py``: Implements early fusion and
         intermediate fusion for ``mmsegmentation`` module.

   -  ``custom/``: external models coming from external repos that we
      are not re-implementing.

      -  ``gto99/`` directory corresponding to *Getting to 99% Accuracy
         in Interactive Segmentation*. Some little modifications were
         conducted (e.g. imports)

         -  ``customized.py``: exposes the gto99 model in standard
            format (see Notes)

   -  ``lightning.py``: ``pytorch_lightning`` implementation of a
      generic model class. Useful for training.

-  ``engine/``: training loops, metrics, and engineering

   -  ``metrics.py``: metrics can be computed from ground truth mask vs
      prediction but also counting the number of clicks is relevant

-  ``tests/``: some tests that check some functions.
-  ``train.py``: script to train a model using components from above
-  ``test.py``: script to evaluate a model using components from above

tasks
-----

-  evaluate a pretrained or not pretrained model over one clicking
   scheme

-  evaluate 99

-  evaluate ritm

-  implement other clicking schemes added:

-  move get_positive_clicks_batch into a robot

-  move smp wrapper inside wrappers

-  use fast jpeg reader instead of opencv

-  profile and reduce training time of full example

--------------

IF THEN ELSE
~~~~~~~~~~~~

*Want to use another dataset?*

Subclass ``SegDataset`` to create a custom loader.

Notes
~~~~~

-  standard model format:
   ``python     def iismodel(x, z, pcs, ncs):         # computations ...         return y``
   ``z`` can be a dict.
-  full mask is different than target region. Full mask refers to the
   original segmentation annotation (can hold multiple classes and
   layers) while target region is a binary mask obtained from full mask.
   There are many possible target regions given a full mask: random
   class, random connected region, merging class or regions, background,
   etc..
-  ritm repo: ``evaluate_model.py`` has the evaluation procedure for the
   model. However, the model is wrapped into a ``predictor`` that
   involves clicking. There are a few predictor classes, for instance
   ``BasePredictor`` will use the model (or net) as
   ``self.net(image_nd, points_nd)['instances']``. The other classes are
   children of this one, so the model is called in the same way (some
   other optimizations may be done).
