from glob import glob
import wfdb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import os
import math
from sklearn import preprocessing



def get_records():

    paths = glob('./dataset/*.atr')

    # Get rid of the extension
    paths = [path[:-4] for path in paths]
    paths.sort()

    return paths


def segmentation(records, type, output_dir=''):


	os.makedirs(output_dir, exist_ok=True)
	results = []
	kernel = np.ones((4, 4), np.uint8)
	count = 1



	for e in tqdm(records):
		signals, fields = wfdb.rdsamp(e, channels = [0])
		signals=preprocessing.scale(np.nan_to_num(signals))
		ann = wfdb.rdann(e, 'atr')

		good = [type]
		ids = np.in1d(ann.symbol, good)
		imp_beats = ann.sample[ids]
		beats = (ann.sample)
		for i in tqdm(imp_beats):
			beats = list(beats)
			j = beats.index(i)
			if(j!=0 and j!=(len(beats)-1)):
				data = (signals[beats[j]-96: beats[j]+96, 0])
				results.append(data)
				plt.plot(data, linewidth=0.5)
				plt.xticks([]), plt.yticks([])
				for spine in plt.gca().spines.values():
					spine.set_visible(False)
				filename = output_dir + 'fig_{}'.format(count) + '.png'
				plt.savefig(filename)
				plt.close()
				im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
				im_gray = cv2.erode(im_gray, kernel, iterations=1)
				im_gray = cv2.resize(im_gray, (128, 128), interpolation=cv2.INTER_LANCZOS4)
				cv2.imwrite(filename, im_gray)
				print('img writtten {}'.format(filename))
				count += 1


	return results



if __name__ == "__main__":
	records = get_records()

	labels = ['N', 'L', 'R', 'A', 'V', '/', 'E', '!']
	output_dirs = ['NOR/', 'LBBB/', 'RBBB/', 'APC/', 'PVC/', 'PAB/', 'VEB/', 'VFE/']

	for type, output_dir in zip(labels, output_dirs):
		sgs = segmentation(records, type, output_dir='./MIT-BIH_AD/'+output_dir)
