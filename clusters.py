

class Clusters(object):


	def __init__(self, data):
		self.__data = data

	def KMeans(self, k=3):
		return [[1.0, 2.3], [1.4, 1.5]]

	def GMM(self, k=3):
		return [[1.4, 1.5], [1.0, 2.3]]

	def data_points(self):
		return self.__data


if __name__ == "__main__":
	data_points = []
	with open('clusters.txt') as f:
		data = f.readlines()
		for line in data:
			data_points.append(list(map(float, line.strip().split(','))))

		# print(len(data_points))

	cluster = Clusters(data_points)
	