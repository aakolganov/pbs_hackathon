import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import argparse
from dataclasses import dataclass
import psutil
import os

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
    Tracks performance metrics and handles collision resolution.
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
        self.last_step_collisions = set()
        
        # Performance Metrics
        self.metrics = {
            'checks': 0,      # Number of distance checks performed
            'time_ms': 0.0,   # Time taken for the physics step
            'memory_mb': 0.0, # Current memory usage
            'flops': 0.0      # Estimated FLOPs
        }
        self.process = psutil.Process(os.getpid())
        
        # Initialize Performance Log File
        self.perf_log_filename = 'performance_metrics.csv'
        with open(self.perf_log_filename, 'w') as f:
            f.write("step,time_ms,memory_mb,checks,flops\n")

    def save_state(self, filename: str) -> None:
        """Saves the current state of all particles to a text file in CSV format."""
        with open(filename, 'w') as f:
            f.write("x_pos,y_pos,x_vel,y_vel,radius,mass\n")
            for i in range(self.config.NUM_PARTICLES):
                f.write(f"{self.positions[i, 0]},{self.positions[i, 1]},"
                        f"{self.velocities[i, 0]},{self.velocities[i, 1]},"
                        f"{self.radii[i]},{self.masses[i]}\n")

    def load_state(self, filename: str) -> None:
        """Loads particle state from a CSV file."""
        positions, velocities, radii, masses = [], [], [], []
        with open(filename, 'r') as f:
            f.readline() # Skip header
            for line in f:
                x_pos, y_pos, x_vel, y_vel, radius, mass = map(float, line.strip().split(','))
                positions.append([x_pos, y_pos])
                velocities.append([x_vel, y_vel])
                radii.append(radius)
                masses.append(mass)

        self.positions = np.array(positions)
        self.velocities = np.array(velocities)
        self.radii = np.array(radii)
        self.masses = np.array(masses)

        if len(self.positions) != self.config.NUM_PARTICLES:
            raise ValueError(f"Particle count mismatch: {len(self.positions)} vs {self.config.NUM_PARTICLES}")

    def initialize_particles(self, seed: int = 42) -> None:
        """Initializes particles with random positions, velocities, radii, and masses."""
        np.random.seed(seed)
        self.radii = np.random.uniform(self.config.MIN_RADIUS, self.config.MAX_RADIUS, self.config.NUM_PARTICLES)
        self.masses = self.radii ** 2
        self.positions = np.random.uniform(
            self.radii[:, None], (self.config.BOX_SIZE - self.radii)[:, None], (self.config.NUM_PARTICLES, 2)
        )
        self.velocities = np.random.uniform(
            self.config.MIN_VELOCITY, self.config.MAX_VELOCITY, (self.config.NUM_PARTICLES, 2)
        )

    def _detect_and_resolve_pair(self, i: int, j: int, current_step_collisions: set) -> None:
        """Helper to resolve a single pair collision."""
        r_diff = self.positions[i] - self.positions[j]
        dist_sq = np.dot(r_diff, r_diff)
        min_dist = self.radii[i] + self.radii[j]
        
        if dist_sq < min_dist**2:
            pair = tuple(sorted((i, j)))
            if pair not in self.last_step_collisions:
                self.resolve_collision(i, j)
            current_step_collisions.add(pair)

    def check_collisions_v1(self) -> None:
        """Brute force collision detection (O(N^2))."""
        for i in range(self.config.NUM_PARTICLES):
            for j in range(i + 1, self.config.NUM_PARTICLES):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < (self.radii[i] + self.radii[j]):
                    self.resolve_collision(i, j)

    def check_collisions_v2(self) -> None:
        """Optimized collision detection preventing 'sticky' collisions."""
        current_step_collisions = set()
        for i in range(self.config.NUM_PARTICLES):
            for j in range(i + 1, self.config.NUM_PARTICLES):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < (self.radii[i] + self.radii[j]):
                    pair = (i, j)
                    if pair not in self.last_step_collisions:
                        self.resolve_collision(i, j)
                    current_step_collisions.add(pair)
        self.last_step_collisions = current_step_collisions

    def check_collisions_v3(self) -> int:
        """Spatial Partitioning (Cell List) implementation. Returns number of checks performed."""
        checks_count = 0
        max_diameter = 2 * self.config.MAX_RADIUS
        n_cells = int(self.config.BOX_SIZE // max_diameter)
        if n_cells < 1: n_cells = 1
        cell_size = self.config.BOX_SIZE / n_cells
        
        grid = {}
        for i in range(self.config.NUM_PARTICLES):
            cx = min(max(int(self.positions[i, 0] // cell_size), 0), n_cells - 1)
            cy = min(max(int(self.positions[i, 1] // cell_size), 0), n_cells - 1)
            if (cx, cy) not in grid: grid[(cx, cy)] = []
            grid[(cx, cy)].append(i)

        current_step_collisions = set()
        neighbor_offsets = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1)]
        
        for (cx, cy), cell_particles in grid.items():
            for dx, dy in neighbor_offsets:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in grid:
                    neighbor_particles = grid[(nx, ny)]
                    if dx == 0 and dy == 0:
                        n = len(cell_particles)
                        checks_count += n * (n - 1) // 2
                        for idx_a in range(n):
                            for idx_b in range(idx_a + 1, n):
                                self._detect_and_resolve_pair(cell_particles[idx_a], cell_particles[idx_b], current_step_collisions)
                    else:
                        checks_count += len(cell_particles) * len(neighbor_particles)
                        for i in cell_particles:
                            for j in neighbor_particles:
                                self._detect_and_resolve_pair(i, j, current_step_collisions)
                                
        self.last_step_collisions = current_step_collisions
        return checks_count

    def resolve_collision(self, i: int, j: int) -> None:
        """Resolves elastic collision between two particles."""
        r_rel_ij = self.positions[i] - self.positions[j]
        r_rel_ji = self.positions[j] - self.positions[i]
        v_rel_ij = self.velocities[i] - self.velocities[j]
        v_rel_ji = self.velocities[j] - self.velocities[i]
        dist = np.linalg.norm(r_rel_ij)
        if np.dot(v_rel_ij, r_rel_ij) < 0:
            norm_r_ij = r_rel_ij / dist
            norm_r_ji = r_rel_ji / dist
            impulse_i = (2 * self.masses[i] / (self.masses[i] + self.masses[j]) * np.dot(v_rel_ij, norm_r_ij) * norm_r_ij)
            impulse_j = (2 * self.masses[j] / (self.masses[i] + self.masses[j]) * np.dot(norm_r_ji, v_rel_ji) * norm_r_ji)
            self.velocities[i] -= impulse_i
            self.velocities[j] -= impulse_j

    def move(self, step: int) -> None:
        """Performs one simulation step: movement, boundary checks, collision resolution, and metrics logging."""
        t0 = time.perf_counter()
        
        self.positions += self.velocities * self.config.DT

        # Wall collisions
        for i in range(self.config.NUM_PARTICLES):
            for d in range(2):
                if (self.positions[i, d] - self.radii[i] < 0 or self.positions[i, d] + self.radii[i] > self.config.BOX_SIZE):
                    self.velocities[i, d] *= -1

        checks = 0
        if self.config.COLLISION_VERSION == 1:
            self.check_collisions_v1()
            checks = self.config.NUM_PARTICLES * (self.config.NUM_PARTICLES - 1) // 2
        elif self.config.COLLISION_VERSION == 2:
            self.check_collisions_v2()
            checks = self.config.NUM_PARTICLES * (self.config.NUM_PARTICLES - 1) // 2
        elif self.config.COLLISION_VERSION == 3:
            checks = self.check_collisions_v3()
        else:
            print(f'Error! Unknown implementation: {self.config.COLLISION_VERSION}')
            exit(1)
        
        dt = time.perf_counter() - t0
        
        # Update Metrics
        self.metrics['time_ms'] = dt * 1000
        self.metrics['checks'] = checks
        self.metrics['flops'] = checks * 15 + (self.config.NUM_PARTICLES * 10)
        self.metrics['memory_mb'] = self.process.memory_info().rss / (1024 * 1024) 
        self.total_elapsed_time += dt

        # Log to CSV
        with open(self.perf_log_filename, 'a') as f:
            f.write(f"{step},{self.metrics['time_ms']:.4f},{self.metrics['memory_mb']:.2f},{checks},{self.metrics['flops']}\n")
        
        # Print to console periodically
        if step % 10 == 0:
            print(f"\rStep {step}: {self.metrics['time_ms']:.2f}ms, {checks} checks, {self.metrics['memory_mb']:.1f}MB", end="")

        if self.config.WRITE_STATE:
            if step % 1 == 0 and self.file_counter < self.config.FILES_LIMIT:
                self.file_counter += 1
                self.save_state(f'simulation_state_step_{step}.csv')

    def update_animation(self, step: int, text_artist):
        """Animation update function that also updates the performance text overlay."""
        self.move(step)
        self.particles.set_offsets(self.positions)
        
        info_str = (
            f"Step: {step}\n"
            f"Particles: {self.config.NUM_PARTICLES}\n"
            f"Time: {self.metrics['time_ms']:.1f} ms\n"
            f"Memory: {self.metrics['memory_mb']:.1f} MB\n"
            f"Pair Checks: {self.metrics['checks']:,}\n"
            f"Est. FLOPs: {self.metrics['flops']/1e6:.2f} M"
        )
        text_artist.set_text(info_str)
        return self.particles, text_artist

    def run_simulation(self, num_steps: int, animate: bool = False) -> None:
        """Runs the simulation, either with animation or in headless mode."""
        if animate:
            # REMOVED explicit backend setting to allow automatic detection
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(0, self.config.BOX_SIZE)
            ax.set_ylim(0, self.config.BOX_SIZE)
            
            # Performance Stats Text (Top Left)
            perf_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left', 
                                family='monospace', fontsize=10,
                                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            self.particles = ax.scatter(
                self.positions[:, 0], self.positions[:, 1],
                s=(self.radii * self.config.SCATTER_SCALE) ** 2, alpha=0.6
            )
            
            ani = animation.FuncAnimation(
                fig, self.update_animation, 
                fargs=(perf_text,),  # Pass the text artist here
                frames=num_steps, interval=1, blit=True, repeat=self.config.REPEAT_ANIMATION
            )
            plt.show()
        else:
            for step in range(num_steps):
                self.move(step)

        print(f'\nTotal elapsed time: {self.total_elapsed_time:.4f} seconds')


def main():
    parser = argparse.ArgumentParser(description="Particle collision simulation")
    parser.add_argument('--version', type=str, required=True, help='collision function version: 1, 2, or 3')
    parser.add_argument('--animate', type=str, required=True, help='1 to animate, 0 otherwise')
    args = parser.parse_args()

    config = SimulationConfig()
    config.COLLISION_VERSION = int(args.version)
    config.WRITE_STATE = True # Set to False to save disk space if needed
    config.REPEAT_ANIMATION = False
    
    simulation = ParticleSimulation(config)
    simulation.initialize_particles()

    print(f'Number of particles: {config.NUM_PARTICLES}')
    print(f'Collision version: {config.COLLISION_VERSION}')
    print(f'Logging performance to: {simulation.perf_log_filename}')

    simulation.run_simulation(num_steps=200, animate=args.animate == '1')


if __name__ == '__main__':
    main()
