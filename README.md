# Overview

This repo contains code for the
["TensorFlow for poets 2" codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2).

Original Repo:
[Github repo: tensorflow-for-poets-2](https://github.com/googlecodelabs/tensorflow-for-poets-2)

This repo contains a simplified and trimmed down version of tensorflow's
[android image classification example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android)
in the `android/` directory.

The `scripts` directory contains helpers for the codelab.

# Instructions

Create a conda environment with Python 3.6 and Tensorflow 1.3 on it. Also install Jupyter (optional) in case
we'd like to use Jupyter Notebook for quick and dirty experiments. Then activate that conda environment:

```bash
source activate
conda create --name py36-tf13 python=3.6 tensorflow=1.3 jupyter
source activate py36-tf13 
```

Navigate to the root directory (i.e. this project directory).

Export environmental variables:

```bash
export IMAGE_SIZE=224
export ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"
```

Start tensorboard:

```bash
tensorboard --logdir tf_files/training_summaries --host=localhost &
```

Access tensorboard at [http://localhost:6006/](http://localhost:6006/)

Run training (change `how_many_training_steps` as you like):

```bash
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos
```

Note the creation of the retrained model:

- `tf_files/retrained_graph.pb`, which contains a version of the selected network with a final layer retrained on your categories.
- `tf_files/retrained_labels.txt`, which is a text file containing labels.

Inspect the `label_image.py` script:

```bash
python -m  scripts.label_image -h
```

Use the Retrained model to do image classification:

```bash
python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb \
    --image=tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg
```

Should get something like this:

```
daisy 0.961026
dandelion 0.0227085
sunflowers 0.0160663
roses 0.000198295
tulips 1.20007e-06
```

Try another one:

```bash
python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/flower_photos/roses/2414954629_3708a1a04d.jpg 
```

Should get something like this:

```bash
roses 0.994535
tulips 0.00540921
dandelion 4.29636e-05
sunflowers 9.80302e-06
daisy 3.06666e-06
```

Try a different learning rate (0.5), and create a new summary (see `summaries_dir`)

```bash
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}_LR_0.5" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos \
  --learning_rate=0.5 \
```

Note that in Tensorboard, our original model (with default learning rate 0.01) trains much better.
This new model (with learning rate 0.5) has persistent high entropy loss.

# Handy references

- [Inception paper](http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)
- [On-device machine learning: TensorFlow on Android (Google Cloud Next '17)](https://www.youtube.com/watch?v=EnFyneRScQ8&feature=youtu.be&t=4m17s)