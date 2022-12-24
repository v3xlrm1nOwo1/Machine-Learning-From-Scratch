import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
	distance =  np.sqrt(np.sum((x2 - x1) ** 2))
	print(f">>> Distance: {distance}")
	return distance


class KNN:
	def __init__(self, K=3):
		self.K = K
		self.gg = 1


	def fit(self, x, y):
		self.x_train = x
		self.y_train = y

	def predict(self, x):
		predictions = [self._predict(i) for i in x] 
		return predictions

	def _predict(self, x):

		# compute the distance
		distances = [euclidean_distance(x, x_train) for x_train in self.x_train]
		print(f">>> Distances: {distances}")


		# get the closest k
		K_indices = np.argsort(distances)[:self.K]
		print(f">>> K_indices: {np.sort(distances)[:self.K]}\n>>> index : {K_indices}")
		K_nearst_labels = [self.y_train[i] for i in K_indices]
		print(f">>> K_nearst_labels: {K_nearst_labels}")

		# majority voye
		most_common = Counter(K_nearst_labels).most_common()
		print(f">>> most_common: {most_common[0][0]}")
		print(f">>> GG: {self.gg}\n{'=' * 50}")
		self.gg += 1

		return most_common[0][0]
	
