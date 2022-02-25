import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import dist
from itertools import repeat
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


class Particle:
    """
    Class to bundle the information of a particle
    """
    def __init__(self, v, pos, best_pos, cur_fit, best_fit):
        self._velocity = v
        self._position = pos
        self._best_pos = best_pos
        self._cur_fit = cur_fit
        self._best_fit = best_fit

    def __str__(self):
        return 'current velocity: {:.3f}, current position: {}, best position: {}, current fitness: {}, ' \
               'and best fitness: {}'\
            .format(self._velocity, self._position, self._best_pos, self._cur_fit, self._best_fit)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velo):
        self._velocity = velo

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos):
        self._position = pos

    @property
    def best_pos(self):
        return self._best_pos

    @best_pos.setter
    def best_pos(self, pos):
        self._best_pos = pos

    @property
    def cur_fit(self):
        return self._cur_fit

    @cur_fit.setter
    def cur_fit(self, fit):
        self._cur_fit = fit

    @property
    def best_fit(self):
        return self._best_fit

    @best_fit.setter
    def best_fit(self, fit):
        self._best_fit = fit


def plot_confusion_matrix(true_classes, predicted):
    """
    Plot confusion matrix
    :param true_classes: true cluster assignments
    :param predicted: predicted cluster assignments
    """
    cm = confusion_matrix(true_classes, predicted)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           ylabel="True label",
           xlabel="Predicted label")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, "{}".format(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


def compare_scatters(data, predicted_classes, true_classes):
    """
    Plot the scatters of the predicted class assignments with the actual class assignments
    :param data: data vectors
    :param predicted_classes: predicted assignment of the data vectors
    :param true_classes: actual assignment of the data vectors
    """
    fig, ax = plt.subplots(2, 1, figsize=(9, 9))
    for class_index in range(2):
        data_class = data[predicted_classes == class_index]
        ax[0].scatter(data_class[:, 0], data_class[:, 1])
        ax[0].set_title('predicted clusters')
        data_class = data[true_classes == class_index]
        ax[1].scatter(data_class[:, 0], data_class[:, 1])
        ax[1].set_title('actual clusters')
    fig.legend(labels=[0, 1], loc=1)
    plt.show()


def calculate_fitness(data_vectors, classes, centroids, n_cluster):
    """
    Calculate fitness of particles centroid positions
    :param data_vectors: the vectors that were assigned to a cluster
    :param classes: which cluster the vectors are assigned to
    :param centroids: coordinates of the centroids
    :param n_cluster: the number of clusters
    :return: fitness according to the equation in the paper
    """
    fitness = 0
    for class_nr in range(n_cluster):
        frequency = len(classes[classes == class_nr])
        part_fit = 0
        for vector in data_vectors:
            part_fit += dist(vector, centroids[class_nr])
        fitness += part_fit/(frequency+1e-6)
    return fitness/n_cluster


def predict_classes(data, pos):
    """
    Predict which vector belongs to which cluster
    :param data: all data points
    :param pos: centroid positions
    :return: which vector is assigned to which cluster
    """
    predicted_cluster = np.zeros(data.shape[0], dtype=np.int32)
    for vector_nr, vector in enumerate(data):
        distances = np.array(list(map(dist, repeat(vector), pos)))
        predicted_cluster[vector_nr] = np.argmin(distances)
    return predicted_cluster


def create_artificial():
    """
    Create the artificial dataset as described in the paper.
    :return: artificial dataset 1
    """
    classes = np.zeros(400, dtype=np.int32)
    z = np.empty((400, 2))
    for v_number in range(400):
        z1, z2 = np.random.uniform(-1, 1, 2)
        z[v_number] = [z1, z2]
        if z1 >= 0.7 or (z1 <= 0.3 and z2 >= -0.2 - z1):
            classes[v_number] = 1
        else:
            classes[v_number] = 0
    return z, classes


def create_particles(data, n_particles, n_classes, n_points):
    """
    Create n_particles instantiating them by sampling random centroids from the data and 0 velocity
    :param data: data to sample centroids from
    :param n_particles: how many particles must be created
    :param n_classes: number of clusters centroids that need to be instantiated
    :param n_points: how many vectors of data are present
    :return: created particles
    """
    particles = np.empty(n_particles, dtype=Particle)
    for part in range(n_particles):
        centroid_indices = np.random.choice(n_points, n_classes, False)
        centroids = data[centroid_indices]
        predicted_classes = predict_classes(data, centroids)
        init_fit = calculate_fitness(data, predicted_classes, centroids, n_classes)
        velocities = np.zeros((n_classes, data.shape[1]), dtype=np.int32)
        particles[part] = Particle(velocities, centroids, centroids, init_fit, init_fit)
    return particles


def pso_algorithm(swarm, data, n_cluster):
    """
    Function that handles all necessary steps for the PSO algorithm described in the paper
    :param swarm: the swarm containing all particles
    :param data: the data which needs to be clustered
    :param n_cluster: the number of clusters
    :return: the best fitness over time and the final best centroid locations
    """
    global_best_fit = swarm[0].best_fit
    global_best_pos = swarm[0].best_pos
    keep_best_fit = np.zeros(100)
    for iteration in range(100):
        for i in range(len(swarm)):
            particle = swarm[i]
            predicted_cluster = predict_classes(data, particle.position)
            # Compute fitness
            fitness = calculate_fitness(data, predicted_cluster, particle.position, n_cluster)
            particle.cur_fit = fitness  # For efficiency purposes, save it in the particle
            # Update local best
            if fitness < particle.best_fit:
                particle.best_fit = fitness
                particle.best_pos = particle.position
        # Update global best
        for i in range(len(swarm)):
            particle = swarm[i]
            fitness = particle.cur_fit
            if fitness < global_best_fit:
                global_best_fit = fitness
                global_best_pos = particle.position
        for i in range(len(swarm)):
            particle = swarm[i]
            # Update velocities and positions
            r1 = np.random.uniform(size=data.shape[1])
            r2 = np.random.uniform(size=data.shape[1])
            velo = particle.velocity
            pos = particle.position
            velo = velo * omega + c1 * r1 * (particle.best_pos - pos) + c2 * r2 * (global_best_pos - pos)
            pos += velo
            particle.velocity = velo
            particle.position = pos
        keep_best_fit[iteration] = global_best_fit
    return keep_best_fit, global_best_pos


def kmeans(data, n_initialisations, n_cluster):
    all_centroids = np.empty((n_initialisations, n_cluster, data.shape[1]))
    for i in range(n_initialisations):
        centroid_indices = np.random.choice(data.shape[0], n_cluster, False)
        centroids = data[centroid_indices]
        all_centroids[i] = centroids
    for i in range(all_centroids.shape[0]):
        current_centroid = all_centroids[i]
        for vector in data:
            pass # calculate distances
    pass


def main(artificial, n_runs):
    """
    Run required functions
    :param artificial: which dataset to use
    :param n_runs: number of runs the code will do
    """
    if artificial:
        data_points, true_classes = create_artificial()
    else:
        data_points = iris_data.iloc[:, :4].to_numpy()
        true_classes = iris_data.iloc[:, 4].to_numpy()
    n_c = len(np.unique(true_classes))
    end_positions = np.zeros((n_runs, n_c, data_points.shape[1]))
    end_fit = np.zeros(n_runs)
    for run in range(n_runs):
        particle_swarm = create_particles(data_points, 10, n_c, data_points.shape[0])
        best_fitness, end_pos = pso_algorithm(particle_swarm, data_points, n_c)
        end_positions[run] = end_pos
        end_fit[run] = best_fitness[-1]
    overall_best_fit_id = np.argmin(end_fit)
    end_pos = end_positions[overall_best_fit_id]
    predictions = predict_classes(data_points, end_pos)
    if artificial:
        compare_scatters(data_points, predictions, true_classes)
    plot_confusion_matrix(true_classes, predictions)
    # Also needs to run for 30 trials, add later
    # kmeans(data_points, 10, n_c)
    kmeans_t = KMeans(n_c, init='random', n_init=10, max_iter=100).fit(data_points)
    predictions = predict_classes(data_points, kmeans_t.cluster_centers_)
    if artificial:
        compare_scatters(data_points, predictions, true_classes)
    plot_confusion_matrix(predictions, true_classes)


# Parameters as stated in paper
omega = 0.72
c1 = 1.49
c2 = 1.49
iris_data = pd.read_csv('iris.data', sep=",", header=None)
dict_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris_data.iloc[:, 4] = iris_data.iloc[:, 4].map(dict_map)
iris_data = iris_data.sample(frac=1).reset_index(drop=True)
main(0, 1)
