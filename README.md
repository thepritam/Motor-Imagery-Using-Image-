## <h1>Motor-Imagery-Using-Imagery</h1>
## Dataset
https://physionet.org/content/eegmmidb/1.0.0/

## Requirements

To install the required dependencies run the following using the Command Prompt:

`pip install -r requirements.txt`

## Preprocessing
```
RUN Elliptic.mlx 
```

Divide the dataset based on the tasks performed in your EEG dataset (In our Case it consists of 4 different tasks: Left Hand Movement, Right Hand Movement,Both Feet Movement,Both Fist Movement) and run the below code. 
```
RUN rearrangement+slidingwindow.py
```
Run the Below code to merge the different dataset which has been pre-processed to train the model.
```
RUN storing-new.py
```

## Training the Model and Visualize Result

```
RUN stacked-ensemble.py
```

```
@article{Biswas2021motor,
  title={Development of Stacked Neural Architecture for Motor Imagery Prediction}
  author={Biswas, Snigdha and Saha, Pritam and Shivam, Shivam and Ekatpure, Purva and Ghosh, Saptarshi}
  }
```
