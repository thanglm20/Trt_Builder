#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# python3 batchPrepare.py --inDir /media/thanglm/ASILLA/Datasets/TestImages/ --outDir ./
# python batchPrepare.py --inDir D:\ThangLM\CalibFiles\calib_train\ --outDir D:\ThangLM\CalibFiles\batch_calib_train

import argparse
import numpy as np
import sys
import os
import glob
import shutil
import struct
from random import shuffle
import cv2
import preprocess

height = 257
width = 449
channel = 3
NUM_BATCHES = 0
NUM_PER_BATCH = 1
NUM_CALIBRATION_IMAGES = 2000

parser = argparse.ArgumentParser()
parser.add_argument('--inDir', required=True, help='Input directory')
parser.add_argument('--outDir', required=True, help='Output directory')

args = parser.parse_args()

CALIBRATION_DATASET_LOC = args.inDir + '*.jpg'


# images to test
imgs = []
print("Location of dataset = " + CALIBRATION_DATASET_LOC)
imgs = glob.glob(CALIBRATION_DATASET_LOC)
shuffle(imgs)
imgs = imgs[:NUM_CALIBRATION_IMAGES]
NUM_BATCHES = NUM_CALIBRATION_IMAGES // NUM_PER_BATCH

print("Image dir: ", args.inDir)
print("Total number of images = " + str(len(imgs)))
print("NUM_PER_BATCH = " + str(NUM_PER_BATCH))
print("NUM_BATCHES = " + str(NUM_BATCHES))
print("NUM IMAGES = " + str(len(imgs)))

# output
outDir  = args.outDir
if os.path.exists(outDir):
    for f in os.listdir(outDir):
        os.remove(os.path.join(outDir, f))
# prepare output
else:
	os.makedirs(outDir)

# load image, switch to RGB, subtract mean, and make dims C x H x W
img = 0
for i in range(NUM_BATCHES):
	batchfile = outDir + "\\batch_calibration" + str(i) + ".batch"
	batchlistfile = outDir + "\\batch_calibration" + str(i) + ".list"
	batchlist = open(batchlistfile,'a')
	batch = np.zeros(shape=(NUM_PER_BATCH, channel, height, width), dtype = np.float32)
	for j in range(NUM_PER_BATCH):
		print("Image: ", img , "/" ,len(imgs) ,": ", imgs[img])
		batchlist.write(os.path.basename(imgs[img]) + '\n')
		image = cv2.imread(imgs[img])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		in_, scale, pad = preprocess.preprocess_factory(image, (height, width))
		in_ = np.array(in_, dtype=np.float32, order='C')
		in_ /= 255.0
		in_ -= np.array((0.485, 0.456, 0.406))
		in_ /= np.array((0.229, 0.224, 0.225))
		in_ = in_.transpose((2,0,1))
		batch[j] = in_
		img += 1

	# save
	batch.tofile(batchfile)
	batchlist.close()

	# Prepend batch shape information
	ba = bytearray(struct.pack("4i", batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]))

	with open(batchfile, 'rb+') as f:
		content = f.read()
		f.seek(0,0)
		f.write(ba)
		f.write(content)


