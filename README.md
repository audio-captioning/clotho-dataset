# Clotho dataset code

Welcome to the repository of Clotho dataset. 

Here you can find the code that will create numpy files with
input/output values, for using Clotho dataset with your audio 
captioning methods. 

The creation of clotho dataset is presented in our paper: 

_K. Drossos, S. Lipping, and T. Virtanen, "Clotho: An Audio Captioning Dataset"_

which is submitted for review to ICASSP 2020. An online version of our paper can 
be found at: https://arxiv.org/abs/1910.09387 

**If you use Clotho dataset, please cite our paper.**

----

## Table of contents

1. [Set up of data and code](#set-up-of-data-and-code)
2. [Using the code](#using-the-code)
3. [Using your own feature extraction functions](#using-your-own-feature-extraction-functions)

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

Then, you just use the script `clotho-dataset-script.sh` and everything will be done. 

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
provide your own function for extraction features. This function should: 

1. Accept as first argument an `numpy.ndarray` object
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