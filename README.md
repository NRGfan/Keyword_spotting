# Keyword_spotting
This repo consists of a few working files for generating training and testing data for the Google Speech Commands Dataset (download it here), testing of various permuatations of feature extraction techniques and model architectures and hyperparameter tuning for the best performing models.

# Environment Setup
The software/packages we are using:
    - Python 3.10.9
    - Numpy 1.23.5
    - Torch 1.13.1
    - Torchaudio 0.13.1
    
Also, install cuda by following instructions from the link [here](https://pytorch.org/get-started/locally/)

You are suggested to use `virtualenv` or `conda` to better setup the run environment. 

You can download the Google Speech Commands dataset v0.01 [here](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz). Then you need to prepare the dataset using the scripts under `prep_data/*`. In short, it will create the full audio list (in `.txt`), training list (in `.csv`) and testing list (in `.csv`). 

You will also need to download the _test set_ of the GSCD dataset to obtain the **silence class**. The link is [here](http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz). Move the folder named `_silence_` from this dataset into the dataset mentioned above (speech_commands_v0.01).

## Preparing dataset

Following (in sequential) is the steps to setup the dataset.

`gen_audio_list.py`: The main idea is to prepare a complete audio list (in `.txt`), which can then be used to generate the train and test datasets. The master list records the location of the audio files and class labels.

`gen_noise_audio.py`: The 1-sec long background noise audio is extracted from the file in the `_background_noise_` folder, as the original audio is very long. 

The audio with silence class is directly copied from the `_silence_` of test set of GSCD dataset.

`prep_for_torch.py`: This is to generate the `.csv` version for the training, validation and testing list.
    
## Neural Network training flow
    
    1. Prepare dataset
    2. Perform data extraction (e.g generate spectrogram, MFCC, Melspectrogram) from dataset look at the section **Generating training\ testing data**
    3. Train the neural network look at the section **Testing each combination of feature extraction technique and model architecture**
    4. If applicable: Tune the hyperparameters look at the section **Hyperparameter tuning with Ray Tune**
    5. If applicable: Evaluate performance of the tuned model to compare with baseline model look at the section **Evaluating model performance**

### Generating training/ testing data 
The files required for generation of training/ testing data is as recorded below and are in the folder :
    1. `MFCC_gen.py`
    2. `Noisy_MFCC_gen.py`
    3. `Melspectrogram_gen.py`
    4. `Noisy_Melspectrogram_gen.py`
    5. `Spectrogram_gen.py`
    6. `Noisy_Spectrogram_gen.py`
    7. `upsampling_MFCC_gen.py`
    8. `LFCC_gen.py`
    9. `MFCC_noisereduced_gen.py`
    10. `25MFCC_gen.py`

The scripts work in a similar fashion. First, they read the train and test csv files previously generated from the Google Speech Commands Dataset (GSCD). They then load the individual audio files in GSCD using the `torchaudio` library, then it generate the datapoints using the respective feature extraction techniques (MFCC, Melspectrogram and Spectrogram).

Files 4, 5 and 6 in the above list additionally introduce noise into the datapoint. They add in white noise to each datapoint, in order to simulate noisy environment when the model may be deployed.

### Testing each combination of feature extraction technique and model architecture
The files required for testing each combination of feature extraction technique and model architecture are as recorded below and are found in the folder `tests`:
    1. `CNN_Melspec_Clean.py`
    2. `CNN_Melspec_Noisy.py`
    3. `CNN_MFCC_Clean.py`
    4. `CNN_MFCC_Noisy.py`
    5. `CNN_Spec_Clean.py`
    6. `CNN_Spec_Noisy.py`
    7. `LSTM_Melspec_Clean.py`
    8. `LSTM_Melspec_Noisy.py`
    9. `LSTM_MFCC_Clean.py`
    10. `LSTM_MFCC_Noisy.py`
    11. `LSTM_Spec_Clean.py`
    12. `LSTM_Spec_Noisy.py`
    13. `MLP_Melspec_Clean.py`
    14. `MLP_Melspec_Noisy.py`
    15. `MLP_MFCC_Clean.py`
    16. `MLP_MFCC_Noisy.py`
    17. `MLP_Spec_Clean.py`
    18. `MLP_Spec_Noisy.py`

The table below shows the various combinations of feature extraction techniques and model architectures tested. The three feature extraction techniques tested are Spectrogram, Melspectrogram and Mel Frequency Cepstral Coeffecients (MFCC), and the three model types tested are Multi Layer Perceptron (MLP), Convolutional Neural Network (CNN) and Long Short Term Memory (LSTM).

| MFCC + MLP | Melspectrogram + MLP | Spectrogram + MLP |
| MFCC + CNN | Melspectrogram + CNN | Spectrogram + CNN |
| MFCC + LSTM | Melspectrogram + LSTM | Spectrogram + LSTM |

On top of that, for each combination of feature extraction and model type, the model is also trained with additional background noise.

**Expected results after 50 epochs of training**

Clean Dataset:
| Combination | Max Validation Accuracy (%) |
| ----------- | --------------------------- |
| CNN + Spectrogram | 78.06 |
| CNN + Melspectrogram | 87.40 |
| CNN + MFCC | 93.52 |
| MLP + Spectrogram | 68.76 |
| MLP + Mel Spectrogram | 72.21 |
| MLP + MFCC | 86.42 |
| LSTM + Spectrogram | 82.14 |
| LSTM + Mel Spectrogram | 89.97 |
| LSTM + MFCC | 93.55 |

Noisy Dataset:
| Combination | Max Validation Accuracy (%) |
| ----------- | --------------------------- |
| CNN + Spectrogram | 69.28 |
| CNN + Melspectrogram | 71.16 |
| CNN + MFCC | 90.83 |
| MLP + Spectrogram | 67.15 |
| MLP + Mel Spectrogram | 69.85 |
| MLP + MFCC | 85.72 |
| LSTM + Spectrogram | 68.03 |
| LSTM + Mel Spectrogram | 67.68 |
| LSTM + MFCC | 91.94 |

### Hyperparameter tuning with Ray Tune
The files required for hyperparameter tuning are as recorded below:
    1. `CustomDataLoader.py`
    2. `Tuning_CNN_MFCC.py`
    3. `Tuning_LSTM_MFCC.py`

After conducting all the tests in the previous step, we want to find the optimal hyperparameters in the model to allow for best performance of the model. I have only implemented hyperparameter tuning for the models in with clean data i.e. no background noise injected. 

We are using the library Ray Tune to implement hyperparameter tuning, it has useful features for hyperparameter tuning. 

`CustomDataSet.py` is a python file containing multiple classes for loading the training and testing data points, which can then be imported in files 2, 3 and 4 in the list. This is because local classes cannot be pickled, giving an error when trying to implement hyperparameter tuning with ray tune.

In files 2, 3 and 4, the search space is defined in the dictionary `config`, which specifies the parameters to be tuned and their respective search spaces.

### Testing LSTM neural network with different feature extraction techniques
We can explore how well certain feature extraction techniques might affect the accuracy:
    1. `upsampling_MFCC_gen.py`: We can resample the audio in GSCD before generating it's MFCC
    2. `LFCC_gen.py`: We can also try using LFCC instead of MFCC which uses linear filterbanks instead of Mel filterbanks to extract features 
    3. `MFCC_noisereduced_gen.py`: We can apply noise reducing algorithm implemented for us in the library `noisereduce` to help remove background noise before generating MFCC
    4. `25MFCC_gen.py`: In this [paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/tje2.12082#:~:text=It%20denotes%20that%20twenty%2Dfive,both%20vowel%20and%20word%20classification.), it suggests that 25 MFCC coefficients provides highest accuracy in audio classification use cases
