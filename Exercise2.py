import numpy as np
import matplotlib.pyplot as plt


class Particle:
    """
    Class to bundle the information of a particle
    """
    def __init__(self, v, pos, best_pos):
        self._velocity = v
        self._position = pos
        self._best_pos = best_pos

    def __str__(self):
        return 'current velocity: {:.3f}, current position: {:.3f} and best position:{:.3f}'\
            .format(self._velocity, self._position, self._best_pos)

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


def swarm_optimize(swarm_pop, parameters):
    """
    Optimize x**2 assuming the swarm only consists of one particle
    :param swarm_pop: the swarm population, in this case only one particle
    :param parameters: omega, alpha and r parameters for this simulation
    :return: the trajectory of the particle
    """
    f = lambda x: x ** 2
    x_best = 0
    omega, alpha, r = parameters
    swarm_positions = np.array([swarm_pop.position])
    while np.abs(swarm_pop.velocity) > 0.01:
        particle = swarm_pop
        velo = particle.velocity
        pos = particle.position
        velo = velo*omega + alpha*r*(particle.best_pos - pos) + alpha*r*(x_best - pos)
        pos += velo
        particle.velocity = velo
        particle.position = pos
        if f(pos) < f(particle.best_pos):
            particle.best_pos = pos

        swarm_positions = np.append(swarm_positions, swarm_pop.position)

    return swarm_positions


v_init = 10
x_init = 20
params = np.array([[0.5, 1.5, 0.5], [0.7, 1.5, 1]])
fix, ax = plt.subplots(2, 1, figsize=(9, 9))
for i in range(2):
    swarm = Particle(v_init, x_init, x_init)
    swarm_pos = swarm_optimize(swarm, params[i])
    ax[i].plot(swarm_pos, color='g')
    ax[i].set_xlabel('iterations')
    ax[i].set_ylabel('x position')
    ax[i].set_title(f'trajectory of a one particle swarm over iterations when omega = {params[i,0]}')
plt.show()
