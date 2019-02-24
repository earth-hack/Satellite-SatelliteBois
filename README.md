# Satellite-SatelliteBois

## Description

`generate_data` will go through the provided dataset and get all slicks. It will also go through the satelite images and pick 10 random patches of size (256, 256). The assumption is that it will not contain any slick (although that is not necessarily the case). The next step is to transform the data to size (256, 256) on slicks and nonslicks (it's strictly not necessary for non-slicks, though). A problem is that aspect ratio is ignored. Probably better performance is achieved if aspect ratio is maintained. Finally, we do an initial shuffle of the dataset and set the correct shape. We make the labels categorical and we add a final channel axis on the features. Keras likes that.

`train_test_split` will just split the dataset into validation, training and test datasets.

`ImageDataGenerator` is used so that we can easily play around with image augmentation.

Next, the model is defined. It's a simple CNN with mostly convolution layers and stides (2, 2). A dense layer with sigmoid activations are added at the end

After some training we do predictions and check for the accuracy.


## Members
- Espen Haugsdal
- Mikael Kvalvær
- Paul Goh
- Musab Almodrra

## Technologies/Software/Libraries Used
- Python
- Keras
- Numpy
- Scikit-Learn

## How to run your code/app/system

To run:
1.  Place the data files in `./data`
2. `python3 src/input_binary.py`

