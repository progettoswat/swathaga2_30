## LipReadingITA
Keras implementation of the method described in the paper 'LipNet: End-to-End Sentence-level Lipreading' by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas (https://arxiv.org/abs/1611.01599).
Reference - https://github.com/rizkiarm/LipNet

## 1. Getting started

- Python 3.10
- [ffmmpeg](https://www.ffmpeg.org)

## 1. Usage

First you need to clone the repository.

Then you can install the package (if you are using an IDE, you can skip this step):
```
cd researchProject/
pip install -e .
```

Next, install the requirements:

```
pip install -r requirements.txt
```
If an error occurs when installing the dlib package, install cmake.




## 2. Dataset ITA
This section will explain the process used to create the Italian dataset. _You can skip this section if you do not want to build a custom dataset._

### 2.1 Sentences
An Italian dataset containing the following sentences was used in this project.

|               Sentence                | ID  |                           Sentence                           | ID  |
|:-------------------------------------:|:---:|:--------------------------------------------------------:|:---:|
|  Salve quanto costa quell' articolo?  |  0  |                   Tutto bene, grazie.                    | 10  |
|     È in offerta, costa 10 euro.      |  1  |                Prendiamo un caffè al bar?                | 11  |
|    Perfetto, vorrei comprarne due.    |  2  |       Certo volentieri, io lo prenderò macchiato.        | 12  |
| Certo ecco a lei, vuole un sacchetto? |  3  |               A che ora arriva il pullman?               | 13  |
|       Sì, grazie e arrivederci.       |  4  |          Dovrebbe arrivare tra qualche minuto.           | 14  |
|     Le auguro una buona giornata.     |  5  |                Quanto costa il biglietto?                | 15  |
|      Buongiorno, io sono Mario.       |  6  | Purtroppo non lo so, però potresti chiedere all’autista. | 16  |
|       Buonasera, io sono Mario.       |  7  |                Va bene, grazie lo stesso.                | 17  |
|       Piacere Luigi, come stai?       |  8  |                          Prego.                          | 18  |
|            Tutto bene, tu?            |  9  |

<br>


### 2.2 Building
To build the dataset, a video recording tool was used (https://github.com/BenedettoSimone/Video-Recorder).
The videos have a size of ``360x288 x 4s``. Use the information provided in the repository and on the main page to replicate this work.
<br><br>After collecting the videos for each subject, the dataset should be organised with the following structure.
```
DATASET:
├───s1
│   ├─── 0-bs.mpg
│   ├─── 1-bs.mpg
│   └───...
├───s2
│   └───...
└───...
    └───...
```
Since ``25fps`` videos are needed, and since the experiment was conducted by recording with several devices, the videos have to be converted to 25fps. To do this, run the ``change_fps/change_fps.py`` script.
Then you have to replace the videos in the ``DATASET`` folder with the newly created videos.

### 2.3 Forced Alignment
For each video, audio and text synchronization (also known as forced alignment) was performed using [Aeneas](https://github.com/readbeyond/aeneas).

After installing Aeneas, create a copy of the dataset and organize the folder as shown below:

```
ForcedAlignment:
│   ├──DatasetCopy:
│       ├───s1
│       │   ├─── 0-bs.mpg
│       │   ├─── 1-bs.mpg
│       │   └───...
│       ├───s2
│       │   └───...
│       └───...
│           └───...
│        
```


Then, follow these steps in the `terminal`:
1. Use the script `alignment/create_fragments_txt.py` to create a `txt` file for each video, following the rules established by Aeneas.
2. Use the script `alignment/autorunAlign.py` to dynamically create the `config` file and generate the `align_json` folder in the `ForcedAlignment` directory.

After running the script, the `ForcedAlignment` folder will have the following structure:

```
ForcedAlignment:
│   ├──DatasetCopy:
│   │   ├───s1
│   │   │   ├─── 0-bs.mpg
│   │   │   ├─── 0-bs.txt
│   │   │   └───...
│   │   ├───s2
│   │       └───...
│   ├──align_json:
│       ├───s1
│       │   ├─── 0-bs.json
│       │   ├─── 1-bs.json
│       │   └───...
│       ├───s2
│       │   └───...   
```


3. Finally, use the script `alignment/alignment_converter.py` to transform each JSON file into an `.align` file with the following format:
```
0 46000 sil
46000 65000 Perfetto
65000 76000 vorrei
76000 88000 comprarne
88000 92000 due
92000 99000 sil
```


The first number indicates the start of that word, and the second number indicates the stop. Each number represents the frame numbers multiplied by 1000 (e.g., frames 0-46 are silence, frames 46-65 are the word "Perfetto," etc).

Now, you will have the `align` folder in the `ForcedAlignment` directory.


### 2.4 Mouth extract
Before starting to extract frames and crop the mouth area insert the ``DATASET`` folder in the project folder and the ``align`` folder in ``Training/datasets/``.

After, execute the script ``MouthExtract/mouth_extract.py`` that return ``100 frames`` for each video in a new folder ``frames``. 

Finally, split this folder in ``Training/datasets/train`` and ``Training/datasets/val`` using 80% for training phase and 20% for validation phase.


## 3. Training
In the previous section, we examined how to build our Italian dataset. Now, we will focus on the training process of the LipNet model.
To train vanilla models use the ``Training/train.py`` script.

If you are using a new dataset, please refer to the FAQ section to change the parameters.

The next table shows the details of the trainings carried out.

| Training            | Details     | Best model |
|---------------------|-------------|--------------------|
| 2023_11_04_16_41_41 | LipNet 16bs | V16_weights598           |

## 4. Prediction
To evaluate the model we used the script ``Predict/predict.py`` placing the video for the prediction phase in the folder ``Predict/PredictVideo``.



