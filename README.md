# Roost Template Matching Exploration
Still very much a work in progress.

## Pipeline

Scripts have values to modify above a comment line separator.

- Run `data-from-folder.py` to generate data from a folder of CIFs if you have them.
- Run `data-prep.py` to generate the training data: pairs of compositions. You can use the CSPML
  data from the URL in that file.
- Run `train.ipynb` to generate a model and save to a checkpoint.
- Use `inference.py` to take a model and run it on new compositions.