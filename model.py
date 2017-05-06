import csv
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, ELU, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import plot_model

PATH_CSV='./data/driving_log.csv'
PATH_IMG_DIR='./data/IMG/'

CORRECTION = 0.2
AUGMENT_TRESHOLD = 0.2

EPOCH=5
BATCH_SIZE=128
LEARNING_RATE=0.001

CROP_IMG_TOP=40
CROP_IMG_BOTTOM=140
NEW_HEIGHT=66
NEW_WIDTH=200

def load_csv():
	lines = []
	with open (PATH_CSV) as csvfile:
		csvfile.readline() # skip header line
		reader = csv.reader(csvfile)

		#After all these loops I will have exact size of my sample data
		for line in reader:  
			for i in range(3):
				img = line[i]
				angle = float(line[3])
				#Flag whether to augment (flip) this image during processing in generator
				toAugment = False
				
				# 0: Center camera image
				# 1: Left camera image
				# 2: Right camera image
				if (i == 1):
					angle += CORRECTION
				elif (i == 2):
					angle -= CORRECTION

				lines.append([img,angle,toAugment])

				#Augment only images from curves to get better steering angles distribution 
				#across all images
				if ((angle >= AUGMENT_TRESHOLD) or (angle <= (-AUGMENT_TRESHOLD))):
					toAugment=True
					lines.append([img,angle,toAugment])

	return lines

def preprocess_img(image, isBGR):
	if (isBGR):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image[CROP_IMG_TOP:CROP_IMG_BOTTOM,:,:]
	image = cv2.resize(image, (NEW_WIDTH,NEW_HEIGHT))
	return image


def generator(samples, batch_size):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			measurements = []

			for batch_sample in batch_samples:

				source_path = batch_sample[0]
				filename = source_path.split('/')[-1]
				current_path = PATH_IMG_DIR + filename
				image = cv2.imread(current_path)
				image = preprocess_img(image, isBGR=True)
				measurement = batch_sample[1]
				toAugment = batch_sample[2]

				if (toAugment):
					images.append(cv2.flip(image,1))
					measurements.append(-1 * measurement)
				else:
					images.append(image)
					measurements.append(measurement)

			X_train = np.array(images)
			y_train = np.array(measurements)

			yield shuffle(X_train, y_train)



def CNN_model():
	model = Sequential()
	model.add(Lambda(lambda x : x / 127.5 - 1, input_shape=(NEW_HEIGHT,NEW_WIDTH,3)))
	model.add(Convolution2D(24,(5,5), strides=2))
	model.add(ELU())
	model.add(Convolution2D(36,(5,5), strides=2))
	model.add(ELU())
	model.add(Convolution2D(48,(5,5), strides=2))
	model.add(ELU())
	model.add(Convolution2D(64,(3,3)))
	model.add(ELU())
	model.add(Convolution2D(64,(3,3)))
	model.add(ELU())
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

def main():
	samples = load_csv()
	train_samples, validation_samples = train_test_split(samples, test_size=0.2)
	print(len(samples))

	train_generator = generator(train_samples, BATCH_SIZE)
	validation_generator = generator(validation_samples, BATCH_SIZE)

	model = CNN_model()

	model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
	history_object = model.fit_generator(train_generator, steps_per_epoch= math.ceil(len(train_samples)/BATCH_SIZE), \
			validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/BATCH_SIZE), \
			epochs=EPOCH, verbose = 1)

	model.save('model.h5')

	### plot the training and validation loss for each epoch
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.savefig('./loss_epochs.png')
	plot_model(model, to_file='./model_shapes.png', show_shapes=True)

if __name__ == '__main__':
    main()
