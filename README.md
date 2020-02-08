# Clotho dataset code

Welcome to the repository of Clotho dataset. 

Here you can find the code that will create numpy files with
input/output values, for using Clotho dataset with your audio 
captioning methods. 

The creation of clotho dataset is presented in our paper: 

_K. Drossos, S. Lipping, and T. Virtanen, "Clotho: An Audio Captioning Dataset," accepted in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), May 4-8, 2020_

An online version of our paper can be found at: https://arxiv.org/abs/1910.09387 

**If you use Clotho dataset, please cite our paper.**

If you use the code in this repository to handle the Clotho dataset, then you can
find useful the `DataLoader` in the
[Clotho data loader repository](https://github.com/audio-captioning/clotho-dataloader).

This repository is maintained by [K. Drossos](https://github.com/dr-costas).

----

## Table of contents

1. [Set up of data and code](#set-up-of-data-and-code)
2. [Using the code](#using-the-code)
3. [Using your own feature extraction functions](#using-your-own-feature-extraction-functions)

----

## Set up of data and code

To set up the data and the code, you need the data of Clotho from Zenodo and the code from the repository. Finally, you need to set up the dependencies. For the latter, we provide a ready made environment with Anaconda. 

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

1. `clotho_audio_development.7z` will produce the `development` directory
2. `clotho_audio_evaluation.7z` will produce the `evaluation` directory

### Clone and set-up the code

To use this repository, you have first to clone it to your computer. Doing so, 
will result in having the top directory of this project `clotho-baseline-dataset`. In
the `clotho-baseline-dataset`, you can find the following directories: 

1. `data`
2. `processes`
3. `settings`
4. `tools`

The directories created by expanding the 7z files mentioned in the previous subsection,
have to be in the `data` directory. That is, the `development` and 
`evaluation` directories have to be placed in the `data` directory. 

If you want to change the naming of the directories of audio data (i.e. `development`
and `evaluation`) then make sure to do the corresponding changes at the 
settings file `dataset_creation.yaml` in the `settings` directory. 

### Install dependencies

To install the necessary dependecies, you can use the Anaconda package manager. If you want to use a different package manager (e.g. PIP) you can check the packages that are needed by the `clotho_conda_requirements.yaml` file. 

With the provide file (), you can create a new environment with conda and use it to run the code in this repository. The name of the environment is `clotho-baseline-dataset`. You can create the environment by using the `conda` commands. 

First, make sure that you have the Anaconda set up in your computer. If not, please install it. Then, open your terminal and navigate to the root directory of the current repository. Then, issue the following command: 

```
conda env create --file clotho_conda_requirements.yaml 
conda activate clotho-baseline-dataset
```

That's it! Now you have installed all the necessary packages, the proper Python version (i.e. 3.7), and you are set up to use the code from this repository. When you are finished, and if youdo not whish to keep the `clotho-baseline-dataset` conda environment, you can use the following command which **will delete the `clotho-baseline-dataset` environment**:

```
conda remove --name clotho-baseline-dataset --all
```

----

## Using the code

The data that you downloaded from Zenodo are audio files and captions in CSV files. To use
the data, you most probably want to have an audio file as an input to your system and a 
caption as a targeted output. Also, most probably you will not use raw audio samples as an input, but some features
extracted from the raw audio. 

With the code in this repository you can:

1. Create numpy objects with audio and corresponding caption
2. Extract features from the audio and insert them to the numpy objects with the captions

To use the code in this repository you can either use the bash script 
`clotho-dataset-script.sh` (which will run the complete process) or use the `main` function
from the files `main.py`, `processes/dataset.py`, and `processes/features.py`. 

### Creating the split data

To create the split data, you can use the settings in the `settings/dataset_creation.yaml`
file and the function `create_dataset` from the `processes/dataset.py` file. You can do this
by either directly using the `processes/dataset.py` file or using the script 
`clotho-dataset-script.sh`. In any case, the process will use the settings in the
`settings/dataset_creation.yaml` file.

### Extracting features 

To extract features from audio data, you have first to create the split data using the 
process described above. Thus, you can follow a two-step approach where you first create 
the split data and then you extract features, or do everything in one step. 

By default, the features that are extracted are 64 log mel-bands, using the settings
that are in the file `settings/feature_extraction.yaml`. 

#### One-step approach
If you do everything in one step, then make sure that both of the entries under `workflow`
in `settings/dataset_creation.yaml` are set to `Yes`. That is, you should have:

````
workflow: 
  create_dataset: Yes
  extract_features: Yes
````

Then, you just use the script `clotho-dataset-script.sh` and everything will be done. To use the script `clotho-dataset-script.sh`, open your terminal, navigate to the root directory of the repository, and issue the following commands: 

```
chmod +x clotho-dataset-script.sh
./clotho-dataset-script.sh
```

Make sure that you have specified correctly the desired/needed names for directories in the 
`settings/dataset_creation.yaml` file.

#### Two-step approach
 
There might be the case where you want to have the data for each split but try 
different features. In such a case, you can first create the data for the splits
and, in a second step, extract the features.

To do so, you can switch the flags in the `settings/dataset_creation.yaml` file:

````
workflow: 
  create_dataset: Yes
  extract_features: Yes
````

and choose the desired action. 

----

## Using your own feature extraction functions

By default, the extracted features are 64 log mel-band energies. You can 
provide your own function for extraction of features. This function should: 

1. Accept as first argument a `numpy.ndarray` object
2. Accept as other arguments the settings for the feature extraction process
3. Return one `numpy.ndarray` object with the extracted features
4. Called `feature_extraction`

The values for settings of the other arguments are from the 
`settings/feature_extraction.yaml` file, under the key `process`. The name of 
the other arguments should be the keys for the entries in the
`process` field.

For example, with the current `settings/feature_extraction.yaml`, to the
feature extraction process are given as arguments all the entries from 
line 19 to 29 of the file. That is: 

````
kwargs = {'sr': 44100, 'nb_fft': 1024, hop_size=512, ...}
````

Finally, you have to specify the package and module of the function in the 
`settings/feature_extraction.yaml` file. The package is specified at the
`package` entry and the module at the `module` entry. As an example, you 
can check the current feature extraction function. 
