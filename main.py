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
        
        # Optimization: specific set to track particle pairs currently in collision
        self.last_step_collisions = set()
        
    def _detect_and_resolve_pair(self, i: int, j: int, current_step_collisions: set) -> None:
        """
        Helper function to check collision between two specific particles (i and j).
        Uses squared distance to avoid costly square root operations for the check.
        """
        # Optimization: Use squared distance to avoid sqrt() unless necessary
        r_diff = self.positions[i] - self.positions[j]
        dist_sq = np.dot(r_diff, r_diff)
        min_dist = self.radii[i] + self.radii[j]
        
        if dist_sq < min_dist**2:
            # Sort indices to ensure unique pair identification (smaller, larger)
            pair = tuple(sorted((i, j)))
            
            # "Sticky" check: Only resolve if they weren't colliding last step
            if pair not in self.last_step_collisions:
                self.resolve_collision(i, j)
            
            # Record this collision so we don't resolve it again next frame if they overlap
            current_step_collisions.add(pair)

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

    def check_collisions_v2(self) -> None:
        """
        Optimized collision detection that prevents 'sticky' collisions.
        
        It maintains a record of overlapping particles from the previous step.
        Collision resolution (bouncing) is only applied if the pair was NOT 
        overlapping in the previous step.
        """
        current_step_collisions = set()
        
        for i in range(self.config.NUM_PARTICLES):
            for j in range(i + 1, self.config.NUM_PARTICLES):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                
                # Check if they are overlapping
                if dist < (self.radii[i] + self.radii[j]):
                    pair = (i, j)
                    
                    # Only resolve if they were not already colliding in the last step
                    if pair not in self.last_step_collisions:
                        self.resolve_collision(i, j)
                    
                    # Add to the set for the next step's memory
                    current_step_collisions.add(pair)
        
        # Update the memory for the next frame
        self.last_step_collisions = current_step_collisions
        
    def check_collisions_v3(self) -> None:
        """
        Spatial Partitioning (Cell List) implementation.
        1. Buckets particles into a grid.
        2. Checks collisions only between particles in the same or adjacent cells.
        """
        # 1. Determine grid size
        # Cell size must be at least the max diameter to guarantee we don't miss 
        # collisions across non-adjacent cells.
        max_diameter = 2 * self.config.MAX_RADIUS
        
        # Number of cells per dimension (ensure at least 1)
        n_cells = int(self.config.BOX_SIZE // max_diameter)
        if n_cells < 1: n_cells = 1
        
        cell_size = self.config.BOX_SIZE / n_cells
        
        # 2. Build the Grid
        # Key: (x_index, y_index), Value: list of particle indices
        grid = {}
        
        for i in range(self.config.NUM_PARTICLES):
            # Calculate grid coordinates
            cx = int(self.positions[i, 0] // cell_size)
            cy = int(self.positions[i, 1] // cell_size)
            
            # Clamp to valid range to handle boundary precision errors
            cx = max(0, min(cx, n_cells - 1))
            cy = max(0, min(cy, n_cells - 1))
            
            if (cx, cy) not in grid:
                grid[(cx, cy)] = []
            grid[(cx, cy)].append(i)

        # 3. Check Collisions
        current_step_collisions = set()
        
        # Neighbors to check: Self + 4 surrounding cells (Half-shell method)
        # This ensures every pair of cells is checked exactly once.
        # Offsets: Center, East, North-East, North, North-West
        neighbor_offsets = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1)]
        
        for (cx, cy), cell_particles in grid.items():
            for dx, dy in neighbor_offsets:
                nx, ny = cx + dx, cy + dy
                
                # If the neighbor cell exists in our populated grid
                if (nx, ny) in grid:
                    neighbor_particles = grid[(nx, ny)]
                    
                    if dx == 0 and dy == 0:
                        # Case A: Within the SAME cell
                        # Check unique pairs: i vs j where j > i
                        for idx_a in range(len(cell_particles)):
                            i = cell_particles[idx_a]
                            for idx_b in range(idx_a + 1, len(cell_particles)):
                                j = cell_particles[idx_b]
                                self._detect_and_resolve_pair(i, j, current_step_collisions)
                    else:
                        # Case B: Between DIFFERENT cells
                        # Check all particles in Cell A vs all in Cell B
                        for i in cell_particles:
                            for j in neighbor_particles:
                                self._detect_and_resolve_pair(i, j, current_step_collisions)
                                
        # Update history for the next step
        self.last_step_collisions = current_step_collisions


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
        
        # --- SELECTION LOGIC ---
        if self.config.COLLISION_VERSION == 1:
            self.check_collisions_v1()
        elif self.config.COLLISION_VERSION == 2:
            self.check_collisions_v2()
        elif self.config.COLLISION_VERSION == 3:
            self.check_collisions_v3()  # <--- NEW CALL
        else:
            print(f'Error! Unknown implementation: {self.config.COLLISION_VERSION}')
            exit(1)
        # -----------------------

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

    simulation.run_simulation(num_steps=10, animate=args.animate == '0')


if __name__ == '__main__':
    main()
