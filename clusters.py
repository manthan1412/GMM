from numpy.random import randint, rand
from numpy.random import random_integers as rand_int
import numpy as np
from math import sqrt, exp

class Clusters(object):


	def __init__(self, data):
		self.__data = data

	def random_centroids(self, k):
		centroids = []
		low = -3
		high = 5
		for i in range(0, k):
			centroids.append([rand()*randint(low, high), rand()*randint(low, high)])
		return centroids

	def distance(self, point, centroid):
		sq_sum = 0
		length = len(centroid)
		for i in range(0, length):
			sq_sum += (point[i]-centroid[i]) ** 2
		return sqrt(sq_sum)

	def get_cluster(self, point, centroids):
		length = len(centroids)
		cluster = 0
		min_dist = self.distance(point, centroids[0])
		for i in range(1, length):
			dist = self.distance(point, centroids[i])
			if dist < min_dist:
				dist = min_dist
				cluster = i
		return cluster

	def calculate_centroids(self, clusters):
		length = len(clusters)
		try:
			dim = len(self.__data[0])
		except:
			dim = 0
		centroids = []
		for i in range(0, length):
			sum_i = [0]*dim
			for point in clusters[i]:
				for j in range(0, dim):
					sum_i[j] += self.__data[point][j]
			n = len(clusters[i])
			if n == 0:
				centroids.append([0.0]*dim)
				continue
			for j in range(0, dim):
				sum_i[j] /= n
			centroids.append(sum_i)
			# print(centroids)
		return centroids

	def should_terminate(self, centroids, new_centroids):
		try:
			return (np.array(centroids) == np.array(new_centroids)).all()
		except:
			return (np.array(centroids) == np.array(new_centroids))

	def KMeans(self, k=3, centroids=None):
		if not centroids:
			centroids = self.random_centroids(k)
		length = len(self.__data)
		clusters = []
		for i in range(0, k):
			clusters.append([])
		for i in range(0, length):
			clusters[self.get_cluster(self.__data[i], centroids)].append(i)
		new_centroids = self.calculate_centroids(clusters)
		print(centroids)
		print(new_centroids)
		if self.should_terminate(centroids, new_centroids):
			return new_centroids
		return self.KMeans(k, new_centroids)

	def compute_mu(self, gamma, k):
		mu = []
		length = len(gamma)
		dim = len(self.__data[0])
		print("--------")
		for i in range(0, k):
			denominator = 0
			numerator = np.array([0.0]*dim)
			for j in range(0, length):
				numerator += gamma[j][i]*np.array(self.__data[j])
				print("numerator: +", gamma[j][i], "*" ,np.array(self.__data[j]))
				denominator += gamma[j][i]
				print("denominator: + ", gamma[j][i])
			mu.append(numerator / denominator)
			print(numerator, denominator)
			print("--------------")
		return np.array(mu)

	def multiply_tr(self, x, mu, dim):
		m = [[0.0 for _ in range(0, dim)] for _ in range(0, dim)]
		diff = np.array(x) - mu
		# print(x)
		# print(mu)
		# print(diff)
		for i in range(0, dim):
			m[i][i] = diff[i]*diff[i]
			for j in range(i+1, dim):
				mul = diff[i]*diff[j]
				m[i][j] = mul
				m[j][i] = mul

		print("2222222222222222222222222")
		print(np.array(m))
		# print(diff)
		# print(diff.reshape(1, 2))
		print(diff.reshape(dim, 1).dot(diff.reshape(1, dim)))
		print("2222222222222222222222222")
		return np.array(m)

	def compute_variance(self, gamma, k, mu):
		variance = []
		length = len(gamma)
		dim = len(self.__data[0])
		print("------------------------------------")
		for i in range(0, k):
			denominator = 0
			numerator = np.array([[0.0 for _ in range(0, dim)] for _ in range(0, dim)])
			for j in range(0, length):
				numerator += gamma[j][i] * self.multiply_tr(self.__data[j], mu[i], dim)
				print("numerator: + ", gamma[j][i], " * " ,self.multiply_tr(self.__data[j], mu[i], dim))
				denominator += gamma[j][i]
				print("denominator: + ", gamma[j][i])
			variance.append(numerator / denominator)
			print(numerator, denominator)
			print("----------------------------------------")
		print(variance)
		return np.array(variance)

	def compute_pi(self, gamma, k):
		pi = []
		length = len(self.__data)
		for i in range(0, k):
			sum_i = 0
			for j in range(0, length):
				sum_i += gamma[j][i]
			pi.append(sum_i/length)
		return np.array(pi)

	# def matmult(self, a, b):
	#     zipped_b = list(zip(*b))
	#     print(a)
	#     print(b)
	#     print(zipped_b)
	#     return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b)) 
 #             for col_b in zipped_b] for row_a in a]

	def compute_gaussian(self, x, mu, pi, variance, k):
		print(x, mu, pi, variance)
		dim = len(x)
		x = np.array(x)
		(sign, logdet) = np.linalg.slogdet(variance)
		det = sign * np.exp(logdet)
		try:
			numerator = x.dot(np.linalg.inv(variance)) #self.matmult(x, np.linalg.inv(variance))
		except:
			return 0.0
		numerator = numerator.dot(x.reshape(dim, 1)) #self.matmult(numerator, x.reshape(dim, 1))
		numerator /= -2
		# print(numerator[0])
		# dummy_d = numerator[0]
		denominator = ((2 *np.pi) ** (dim/2)) * sqrt(abs(det))
		numerator[0] -= np.log(denominator)		
		numerator = np.exp(numerator[0])

		# print(numerator)
		# print(np.exp(dummy_d)/denominator)
		return numerator

	def compute_gamma(self, variance, mu, pi, k):
		gamma = []
		gaussian = []
		length = len(self.__data)
		for i in range(0, length):
			gaus = []
			for j in range(0, k):
				gaus.append(pi[j]*self.compute_gaussian(self.__data[i], mu[j], pi[j], variance[j], k))
			gaussian.append(gaus)

		print(gaussian)
		gaussian = np.array(gaussian)
		for i in range(0, length):
			# gamma_row = []
			gamma.append(gaussian[i]/gaussian[i].sum())
			# sum_row = gaussian[i].sum()
			# for j in range(0, k):
				# gamma_row.append(gaussian[i][j]/sum_row)
		return np.array(gamma)

	def EM(self, mu, variance, pi, gamma, k):
		print(gamma)
		gamma = self.compute_gamma(variance, mu, pi, k)
		print(gamma)
		new_mu = self.compute_mu(gamma, k)
		if self.should_terminate(mu, new_mu):
			return mu
		variance = self.compute_variance(gamma, k, mu)
		pi = self.compute_pi(gamma, k)
		
		t = input()
		return self.EM(new_mu, variance, pi, gamma, k)

	def GMM(self, k=3):
		gamma = []
		low = 0
		high = 50
		length = len(self.__data)
		for i in range(0, length):
			cluster_assignment = rand_int(0, 50, k)
			while cluster_assignment.sum() == 0:
				cluster_assignment = rand_int(0, 50, k)
			gamma.append(cluster_assignment/cluster_assignment.sum())

		# print(gamma)
		mu = self.compute_mu(gamma, k)
		# print(mu)
		variance = self.compute_variance(gamma, k, mu)
		pi = self.compute_pi(gamma, k)
		# print(pi)

		# for i in range(0, k):
			# cluster_mu = self.clusters
		# self.EM(gamma, compute())
		print("*******************")
		print("mu: ", mu)
		print("variance: ", variance)
		print("pi: ", pi)
		print("*******************")
		return self.EM(mu, variance, pi, gamma, k)

	def data_points(self):
		return self.__data



if __name__ == "__main__":
	data_points = []
	with open('clusters_test_gmm.txt') as f:
		data = f.readlines()
		for line in data:
			data_points.append(list(map(float, line.strip().split(','))))

		# print(len(data_points))

	cluster = Clusters(data_points)
	print(cluster.data_points())
	# print(cluster.KMeans())
	print(cluster.GMM())
