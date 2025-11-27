import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import argparse
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    NUM_PARTICLES: int = 1000
    BOX_SIZE: float = 1.0
    DT: float = 0.01
    MIN_RADIUS: float = 0.005
    MAX_RADIUS: float = 0.02
    MIN_VELOCITY: float = -0.1
    MAX_VELOCITY: float = 0.1
    SCATTER_SCALE: float = 1000
    COLLISION_VERSION: int = 1
    REPEAT_ANIMATION: bool = True
    WRITE_STATE: bool = False
    # A very rough estimation for the maximum number of files that can be created
    FILES_LIMIT: int = int(200 * 1024 * 1024 / (150 * NUM_PARTICLES))


class ParticleSimulation:
    """
    Simulates the motion and interaction of particles within a bounded environment.

    This class handles the initialization, movement, and collision resolution of
    particles inside a simulation box. It supports running simulations with or
    without animations, where particles are depicted as moving circles with varying
    radii and velocities. The interactions between particles include collision detection
    and resolution based on physical dynamics principles.

    :ivar config: Configuration settings for the simulation, defining parameters
        such as the number of particles, box size, velocity bounds, and more.
    :type config: SimulationConfig
    :ivar total_elapsed_time: The accumulated time spent processing collisions in seconds.
    :type total_elapsed_time: float
    :ivar radii: The radii of individual particles in the simulation.
    :type radii: numpy.ndarray
    :ivar masses: The masses of individual particles, derived from their radii.
    :type masses: numpy.ndarray
    :ivar positions: The positions of the particles in the 2D simulation space.
    :type positions: numpy.ndarray
    :ivar velocities: The velocities of particles in the 2D simulation space.
    :type velocities: numpy.ndarray
    :ivar particles: Handles the scatter plot representing particles for visualization,
        used only in animated simulations. None if animation is not enabled.
    :type particles: matplotlib.collections.PathCollection or None
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.total_elapsed_time = 0.0
        self.radii = np.zeros(config.NUM_PARTICLES)
        self.masses = np.zeros(config.NUM_PARTICLES)
        self.positions = np.zeros((config.NUM_PARTICLES, 2))
        self.velocities = np.zeros((config.NUM_PARTICLES, 2))
        self.particles = None
        self.file_counter = 0

    def save_state(self, filename: str) -> None:
        """
        Saves the current state of all particles to a text file in CSV format.

        Each row represents one particle with the following columns:
        x_position, y_position, x_velocity, y_velocity, radius, mass

        :param filename: Name of the file to save the state to
        :type filename: str
        """
        with open(filename, 'w') as f:
            # Write header
            f.write("x_pos,y_pos,x_vel,y_vel,radius,mass\n")

            # Write data for each particle
            for i in range(self.config.NUM_PARTICLES):
                f.write(f"{self.positions[i, 0]},{self.positions[i, 1]},"
                        f"{self.velocities[i, 0]},{self.velocities[i, 1]},"
                        f"{self.radii[i]},{self.masses[i]}\n")

    def load_state(self, filename: str) -> None:
        """
        Loads particle state from a CSV file and initializes the simulation with it.

        Expects a file created by save_state() containing comma-separated values
        with columns: x_position, y_position, x_velocity, y_velocity, radius, mass

        :param filename: Name of the file to load the state from
        :type filename: str
        :raises ValueError: If the loaded data doesn't match the configured number of particles
        """
        positions = []
        velocities = []
        radii = []
        masses = []

        with open(filename, 'r') as f:
            # Skip header
            header = f.readline()

            # Read data
            for line in f:
                x_pos, y_pos, x_vel, y_vel, radius, mass = map(float, line.strip().split(','))
                positions.append([x_pos, y_pos])
                velocities.append([x_vel, y_vel])
                radii.append(radius)
                masses.append(mass)

        # Convert to numpy arrays
        self.positions = np.array(positions)
        self.velocities = np.array(velocities)
        self.radii = np.array(radii)
        self.masses = np.array(masses)

        # Verify particle count
        if len(self.positions) != self.config.NUM_PARTICLES:
            raise ValueError(
                f"Loaded data contains {len(self.positions)} particles, "
                f"but {self.config.NUM_PARTICLES} configured"
            )

    def initialize_particles(self, seed: int = 42) -> None:
        """
        Initializes particles with random positions, velocities, radii, and masses based on configuration
        parameters. The method uses the given seed value to ensure reproducibility in the random number
        generation process. Particles' radii, positions and velocities are assigned within specified bounds,
        and masses are calculated based on the radii.

        :param seed: Seed value for the random number generator.
        :type seed: int
        :return: This method does not return any value.
        :rtype: None
        """
        np.random.seed(seed)
        self.radii = np.random.uniform(
            self.config.MIN_RADIUS,
            self.config.MAX_RADIUS,
            self.config.NUM_PARTICLES
        )
        self.masses = self.radii ** 2
        self.positions = np.random.uniform(
            self.radii[:, None],
            (self.config.BOX_SIZE - self.radii)[:, None],
            (self.config.NUM_PARTICLES, 2)
        )
        self.velocities = np.random.uniform(
            self.config.MIN_VELOCITY,
            self.config.MAX_VELOCITY,
            (self.config.NUM_PARTICLES, 2)
        )

    def check_collisions_v1(self) -> None:
        """
        Checks for collisions among particles and resolves them if detected.

        The method iterates through all pairs of particles and computes the distance
        between their positions. If the distance between two particles is less than
        the sum of their radii, it identifies a collision and invokes the
        `resolve_collision` method to handle it.
        """
        for i in range(self.config.NUM_PARTICLES):
            for j in range(i + 1, self.config.NUM_PARTICLES):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < (self.radii[i] + self.radii[j]):
                    self.resolve_collision(i, j)

    def resolve_collision(self, i: int, j: int) -> None:
        """
        Resolves a collision between two objects identified by their indices, i and j.
        This function adjusts the velocities of the objects based on their relative
        positions and velocities. It ensures that the objects bounce off each other
        according to the rules of elastic collisions.

        The collision resolution considers the masses, positions, and velocities
        of the objects. The approach assumes that the objects would not overlap
        post-collision and calculates impulses to adjust their velocities.

        :param i: Index of the first object involved in the collision.
        :param j: Index of the second object involved in the collision.
        :type i: int
        :type j: int
        :return: This function does not return any value.
        :rtype: None
        """
        r_rel_ij = self.positions[i] - self.positions[j]
        r_rel_ji = self.positions[j] - self.positions[i]
        v_rel_ij = self.velocities[i] - self.velocities[j]
        v_rel_ji = self.velocities[j] - self.velocities[i]
        dist = np.linalg.norm(r_rel_ij)
        if np.dot(v_rel_ij, r_rel_ij) < 0:
            norm_r_ij = r_rel_ij / dist
            norm_r_ji = r_rel_ji / dist
            impulse_i = (2 * self.masses[i] /
                       (self.masses[i] + self.masses[j]) * np.dot(v_rel_ij, norm_r_ij) * norm_r_ij)
            impulse_j = (2 * self.masses[j] /
                       (self.masses[i] + self.masses[j]) * np.dot(norm_r_ji, v_rel_ji) * norm_r_ji)
            self.velocities[i] -= impulse_i
            self.velocities[j] -= impulse_j

    def move(self, step: int) -> None:
        """
        Updates the positions of particles and handles wall collisions as part of a
        step in the simulation. Additionally, checks for particle collisions
        and updates the elapsed time of the simulation.

        :param step: The current simulation step number.
        :type step: int
        :return: None
        """
        self.positions += self.velocities * self.config.DT

        for i in range(self.config.NUM_PARTICLES):
            for d in range(2):
                if (self.positions[i, d] - self.radii[i] < 0 or
                        self.positions[i, d] + self.radii[i] > self.config.BOX_SIZE):
                    self.velocities[i, d] *= -1

        start_time = time.time()
        if self.config.COLLISION_VERSION == 1:
            self.check_collisions_v1()
        # elif self.config.COLLISION_VERSION == 2:
        #     self.check_collisions_v2()
        else:
            print(f'Error! Unknown implementation: {self.config.COLLISION_VERSION}')
            exit(1)
        elapsed_time = time.time() - start_time
        self.total_elapsed_time += elapsed_time
        print(f'\nStep: {step}')
        print(f'Elapsed time: {elapsed_time}')

        if self.config.WRITE_STATE:
            if step % 1 == 0 and self.file_counter < self.config.FILES_LIMIT:
                self.file_counter += 1
                self.save_state(f'simulation_state_step_{step}.csv')
            else:
                print(f'Too many files, skipping call for \'save_state\' at step {step}')

    def update_animation(self, step: int):
        """
        Updates the animation state by moving the particles and setting their offsets.
        This function is typically used as an update function in animation loops.
        The positions of the particles are updated based on the given step, and
        the updated positions are applied to the visualization. It ensures that
        particle rendering reflects the most recent positions.

        :param step: The step value determining how much each particle moves.
        :type step: int
        :return: The updated particle artists for rendering.
        :rtype: tuple
        """
        self.move(step)
        self.particles.set_offsets(self.positions)
        return self.particles,

    def run_simulation(self, num_steps: int, animate: bool = False) -> None:
        """
        Run the simulation for a specified number of steps, with an optional animation.

        This method is used to either animate the simulation or run it step-by-step
        without visualization. If animation is enabled, it sets up the plotting
        environment and animates the movement of particles. Otherwise, it proceeds with
        a step-by-step update of the simulation state.

        :param num_steps: The number of steps the simulation should run.
        :type num_steps: int
        :param animate: Whether to animate the simulation. Defaults to False.
        :type animate: bool, optional
        :return: This method does not return anything.
        :rtype: None
        """
        if animate:
            matplotlib.use('QT5Agg')
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(0, self.config.BOX_SIZE)
            ax.set_ylim(0, self.config.BOX_SIZE)
            self.particles = ax.scatter(
                self.positions[:, 0],
                self.positions[:, 1],
                s=(self.radii * self.config.SCATTER_SCALE) ** 2,
                alpha=0.6
            )
            ani = animation.FuncAnimation(
                fig, self.update_animation, frames=num_steps,
                interval=20, blit=True, repeat=self.config.REPEAT_ANIMATION
            )
            plt.show()
        else:
            for step in range(num_steps):
                self.move(step)

        print(f'\nTotal elapsed time: {self.total_elapsed_time} seconds')
        print(f'Average time per step: {self.total_elapsed_time / num_steps} seconds')


def main():
    """
    Main entry point for the particle collision simulation. This function parses
    command-line arguments, initializes the simulation configuration, and starts the
    simulation process. The simulation includes options for enabling animations and
    choosing the version of the collision function implementation.

    :raises SystemExit: If the required arguments are missing or invalid when parsing
        command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Particle collision simulation")
    # Add arguments
    parser.add_argument('--version', type=str, required=True,
                        help='numerical version of the collision function implementation: 1, ...')
    parser.add_argument('--animate', type=str, required=True,
                        help='1 if simulations should be animated, 0 otherwise')
    args = parser.parse_args()

    config = SimulationConfig()
    config.COLLISION_VERSION = int(args.version)
    config.WRITE_STATE = True
    config.REPEAT_ANIMATION = False
    simulation = ParticleSimulation(config)
    simulation.initialize_particles()

    print(f'Number of particles: {config.NUM_PARTICLES}')
    print(f'Box size: {config.BOX_SIZE}')
    print(f'Number of dimensions: 2')
    print(f'Collision version: {config.COLLISION_VERSION}')
    print(f'Write state: {config.WRITE_STATE}')
    print(f'Files limit: {config.FILES_LIMIT}')
    print(f'Repeat animation: {config.REPEAT_ANIMATION}')

    simulation.run_simulation(num_steps=10, animate=args.animate == '1')


if __name__ == '__main__':
    main()
