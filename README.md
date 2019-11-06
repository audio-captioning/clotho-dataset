# Clotho dataset code

Welcome to the repository of Clotho dataset. 

Here you can find the code that will create numpy files with
input/output values, for using Clotho dataset with your audio 
captioning methods. 

If you use Clotho dataset, please cite our paper that can be
found here: https://arxiv.org/abs/1910.09387

----

## Table of contents

1. [Set up of data and code](#set-up-of-data-and-code)
2. [Using the code](#using-the-code)
3. [Using your own feature extraction functions](#using-your-own-feature-extraction-functions)
4. [About Clotho, one of the three fates](#about-clotho-one-of-the-three-fates)

----

## Set up of data and code

To set up the data and the code, you need the data of Clotho dfrom Zenodo and the code from the repository. 

### Getting the data from Zenodo

To start using Clotho dataset, you have first to download it from Zenodo: 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3490684.svg)](https://doi.org/10.5281/zenodo.3490684)

There are at least four files that you need to have from the Zenodo repository, two for the development split and two for
the evaluation split: 

1. `clotho_audio_development.7z`: the 7z archive file with the audio data for the **development** split 
2. `clotho_captions_development.csv`: the CSV file with the captions of the **development** split 
3. `clotho_audio_evaluation.7z`: the 7z archive file with the audio data for the **evaluation** split
4. `clotho_captions_evaluation.csv`: the CSV file with the captions of the **evaluation** split

Expanding the 7z files, will produce two directories: 

1. `clotho_audio_development.7z` will produce the   directory
2. `clotho_audio_evaluation.7z` will produce the   directory

### Clone and set-up the code

To use this repository, you have first to clone it to your computer. Doing so, 
will result in having the top directory of this project `clotho-baseline-dataset`. In
the `clotho-baseline-dataset`, you can find the following directories: 

1. `data`
2. `processes`
3. `settings`
4. `tools`

The directories created by expanding the 7z files mentioned in the previous subsection,
have to be in the `data` directory. That is, the `clotho_audio_development` and 
`clotho_audio_evaluation` directories have to be placed in the `data` directory. 

If you want to change the naming of the directories of audio data (i.e. `clotho_audio_development`
and `clotho_audio_evaluation`) then make sure to do the corresponding changes at the 
settings file `dataset_creation.yaml` in the `settings` directory. 

----

## Using the code

To use the code in this repository you can either use the bash script 
`clotho-dataset-script.sh` (which will run the complete process) or use the `main` function
from the files `main.py`, `processes/dataset.py`, and `processes/features.py`. 

### Creating the split data

### Extracting features 

----

## Using your own feature extraction functions

----

## About Clotho, one of the three fates
