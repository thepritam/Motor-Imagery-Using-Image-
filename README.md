## <h1>Motor-Imagery-Using-Imagery</h1>
## Dataset
https://physionet.org/content/eegmmidb/1.0.0/

## Requirements

To install the required dependencies run the following using the Command Prompt:

`pip install -r requirements.txt`

## Preprocessing
```
RUN Elliptic.mlx 
RUN rearrangement+slidingwindow.py
RUN storing-new.py
```

## Training the Model and Visualize Result

```
RUN stacked-ensemble.py
```
## Citation 
If this repository helps you in your research in any way, please cite our paper:

```
@article{Biswas2021motor,
  title={Development of Stacked Neural Architecture for Motor Imagery Prediction}
  author={Biswas, Snigdha and Saha, Pritam and Shivam, Shivam and Ekatpure, Purva and Ghosh, Saptarshi}
  }
```
