import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import argparse
from dataclasses import dataclass
import psutil
import os
from numba import njit, prange

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
    FILES_LIMIT: int = int(200 * 1024 * 1024 / (150 * NUM_PARTICLES))

# ---------------------------------------------------------
# NUMBA KERNELS (Micro-Optimized)
# ---------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def move_particles_kernel(positions, velocities, radii, box_size, dt):
    """Updates positions and handles wall collisions in parallel."""
    n = len(positions)
    for i in prange(n):
        # Position Update
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt

        # Wall Collisions (Branchless-ish optimization possible, but simple if/else is fine)
        r = radii[i]
        x = positions[i, 0]
        y = positions[i, 1]

        if x - r < 0:
            positions[i, 0] = r
            velocities[i, 0] *= -1
        elif x + r > box_size:
            positions[i, 0] = box_size - r
            velocities[i, 0] *= -1
            
        if y - r < 0:
            positions[i, 1] = r
            velocities[i, 1] *= -1
        elif y + r > box_size:
            positions[i, 1] = box_size - r
            velocities[i, 1] *= -1

@njit(fastmath=True, cache=True)
def check_collisions_v1_numba(positions, velocities, radii, masses):
    """Brute Force - Single Threaded"""
    n = len(positions)
    checks = 0
    for i in range(n):
        for j in range(i + 1, n):
            checks += 1
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist_sq = dx*dx + dy*dy
            
            rad_sum = radii[i] + radii[j]
            if dist_sq < rad_sum * rad_sum:
                # INLINED RESOLUTION
                dvx = velocities[i, 0] - velocities[j, 0]
                dvy = velocities[i, 1] - velocities[j, 1]
                
                if dvx*dx + dvy*dy < 0:
                    dist = np.sqrt(dist_sq)
                    nx = dx / dist
                    ny = dy / dist
                    
                    mass_sum = masses[i] + masses[j]
                    impulse = (2 * (dvx*nx + dvy*ny)) / mass_sum
                    
                    impulse_x = impulse * nx
                    impulse_y = impulse * ny
                    
                    velocities[i, 0] -= impulse_x * masses[j]
                    velocities[i, 1] -= impulse_y * masses[j]
                    velocities[j, 0] += impulse_x * masses[i]
                    velocities[j, 1] += impulse_y * masses[i]
    return checks

@njit(parallel=True, fastmath=True, cache=True)
def check_collisions_v2_numba(positions, velocities, radii, masses):
    """Brute Force - Parallel"""
    n = len(positions)
    # Parallel reduction for checks count is implicit or ignored for speed
    # We return approximate checks count for v2 parallel to avoid atomic overhead
    for i in prange(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist_sq = dx*dx + dy*dy
            
            rad_sum = radii[i] + radii[j]
            if dist_sq < rad_sum * rad_sum:
                # INLINED RESOLUTION
                dvx = velocities[i, 0] - velocities[j, 0]
                dvy = velocities[i, 1] - velocities[j, 1]
                
                if dvx*dx + dvy*dy < 0:
                    dist = np.sqrt(dist_sq)
                    nx = dx / dist
                    ny = dy / dist
                    
                    mass_sum = masses[i] + masses[j]
                    impulse = (2 * (dvx*nx + dvy*ny)) / mass_sum
                    
                    impulse_x = impulse * nx
                    impulse_y = impulse * ny
                    
                    # Note: Simultaneous updates in parallel loops can have race conditions.
                    # This is generally acceptable for visualization physics or dealt with locks.
                    velocities[i, 0] -= impulse_x * masses[j]
                    velocities[i, 1] -= impulse_y * masses[j]
                    velocities[j, 0] += impulse_x * masses[i]
                    velocities[j, 1] += impulse_y * masses[i]
                    
    return n * (n - 1) // 2

@njit(fastmath=True, cache=True)
def check_collisions_v3_numba(positions, velocities, radii, masses, 
                              head, next_particle, cell_size, n_cells_x, n_cells_y):
    """
    Optimized Cell List.
    Uses pre-allocated arrays 'head' and 'next_particle' to avoid memory thrashing.
    """
    n = len(positions)
    
    # 1. Reset Head (Fast fill)
    head.fill(-1)
    
    # 2. Build Linked List
    for i in range(n):
        cx = int(positions[i, 0] / cell_size)
        cy = int(positions[i, 1] / cell_size)
        
        # Fast Clamp
        if cx < 0: cx = 0
        elif cx >= n_cells_x: cx = n_cells_x - 1
        
        if cy < 0: cy = 0
        elif cy >= n_cells_y: cy = n_cells_y - 1
        
        next_particle[i] = head[cx, cy]
        head[cx, cy] = i

    checks = 0
    # Offsets: Self (handled by list order), East, NE, North, NW
    neighbor_dx = np.array([1, 1, 0, -1], dtype=np.int32)
    neighbor_dy = np.array([0, 1, 1, 1], dtype=np.int32)
    
    for cx in range(n_cells_x):
        for cy in range(n_cells_y):
            i = head[cx, cy]
            while i != -1:
                
                # A. Intra-cell collisions
                j = next_particle[i]
                while j != -1:
                    checks += 1
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_sq = dx*dx + dy*dy
                    rad_sum = radii[i] + radii[j]
                    
                    if dist_sq < rad_sum * rad_sum:
                        # INLINED RESOLUTION
                        dvx = velocities[i, 0] - velocities[j, 0]
                        dvy = velocities[i, 1] - velocities[j, 1]
                        if dvx*dx + dvy*dy < 0:
                            dist = np.sqrt(dist_sq)
                            nx = dx / dist; ny = dy / dist
                            mass_sum = masses[i] + masses[j]
                            impulse = (2 * (dvx*nx + dvy*ny)) / mass_sum
                            ix = impulse * nx; iy = impulse * ny
                            velocities[i, 0] -= ix * masses[j]; velocities[i, 1] -= iy * masses[j]
                            velocities[j, 0] += ix * masses[i]; velocities[j, 1] += iy * masses[i]

                    j = next_particle[j]
                
                # B. Inter-cell collisions
                for k in range(4):
                    nx = cx + neighbor_dx[k]
                    ny = cy + neighbor_dy[k]
                    
                    if 0 <= nx < n_cells_x and 0 <= ny < n_cells_y:
                        j = head[nx, ny]
                        while j != -1:
                            checks += 1
                            dx = positions[i, 0] - positions[j, 0]
                            dy = positions[i, 1] - positions[j, 1]
                            dist_sq = dx*dx + dy*dy
                            rad_sum = radii[i] + radii[j]
                            
                            if dist_sq < rad_sum * rad_sum:
                                # INLINED RESOLUTION
                                dvx = velocities[i, 0] - velocities[j, 0]
                                dvy = velocities[i, 1] - velocities[j, 1]
                                if dvx*dx + dvy*dy < 0:
                                    dist = np.sqrt(dist_sq)
                                    nx = dx / dist; ny = dy / dist
                                    mass_sum = masses[i] + masses[j]
                                    impulse = (2 * (dvx*nx + dvy*ny)) / mass_sum
                                    ix = impulse * nx; iy = impulse * ny
                                    velocities[i, 0] -= ix * masses[j]; velocities[i, 1] -= iy * masses[j]
                                    velocities[j, 0] += ix * masses[i]; velocities[j, 1] += iy * masses[i]
                            j = next_particle[j]
                i = next_particle[i]
    return checks

# ---------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------

class ParticleSimulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.total_elapsed_time = 0.0
        
        # Data Arrays
        self.radii = np.zeros(config.NUM_PARTICLES, dtype=np.float64)
        self.masses = np.zeros(config.NUM_PARTICLES, dtype=np.float64)
        self.positions = np.zeros((config.NUM_PARTICLES, 2), dtype=np.float64)
        self.velocities = np.zeros((config.NUM_PARTICLES, 2), dtype=np.float64)
        
        # Pre-calculate Grid Parameters for V3 to avoid re-calc
        self.max_dia = 2 * self.config.MAX_RADIUS
        self.n_cells = int(self.config.BOX_SIZE // self.max_dia)
        if self.n_cells < 1: self.n_cells = 1
        self.cell_size = self.config.BOX_SIZE / self.n_cells
        
        # MEMORY OPTIMIZATION: Pre-allocate Cell List Arrays
        # We use n_cells for both X and Y since box is square
        self.grid_head = np.full((self.n_cells, self.n_cells), -1, dtype=np.int32)
        self.grid_next = np.full(config.NUM_PARTICLES, -1, dtype=np.int32)

        self.particles = None
        self.file_counter = 0
        self.metrics = {'checks': 0, 'time_ms': 0.0, 'memory_mb': 0.0, 'flops': 0.0}
        self.process = psutil.Process(os.getpid())
        
        # IO OPTIMIZATION: Keep file handle open
        self.perf_log_filename = 'performance_check.csv'
        print(f"Writing performance metrics to {self.perf_log_filename}...")
        self.perf_log_file = open(self.perf_log_filename, 'w')
        self.perf_log_file.write("step,time_ms,memory_mb,checks,flops\n")

    def __del__(self):
        """Destructor to ensure file is closed."""
        if hasattr(self, 'perf_log_file') and not self.perf_log_file.closed:
            self.perf_log_file.close()

    def save_state(self, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write("x_pos,y_pos,x_vel,y_vel,radius,mass\n")
            # Optimization: Vectorized string creation is hard in vanilla Python, 
            # but we can't change format. The loop is unavoidable without pandas.
            for i in range(self.config.NUM_PARTICLES):
                f.write(f"{self.positions[i, 0]:.6f},{self.positions[i, 1]:.6f},"
                        f"{self.velocities[i, 0]:.6f},{self.velocities[i, 1]:.6f},"
                        f"{self.radii[i]:.6f},{self.masses[i]:.6f}\n")

    def initialize_particles(self, seed: int = 42):
        np.random.seed(seed)
        self.radii = np.random.uniform(self.config.MIN_RADIUS, self.config.MAX_RADIUS, self.config.NUM_PARTICLES)
        self.masses = self.radii ** 2
        self.positions = np.random.uniform(
            self.radii[:, None], (self.config.BOX_SIZE - self.radii)[:, None], (self.config.NUM_PARTICLES, 2)
        )
        self.velocities = np.random.uniform(
            self.config.MIN_VELOCITY, self.config.MAX_VELOCITY, (self.config.NUM_PARTICLES, 2)
        )
        
        print("Compiling Numba kernels... (Warmup)")
        move_particles_kernel(self.positions, self.velocities, self.radii, self.config.BOX_SIZE, self.config.DT)
        if self.config.COLLISION_VERSION == 3:
             check_collisions_v3_numba(
                 self.positions, self.velocities, self.radii, self.masses,
                 self.grid_head, self.grid_next, self.cell_size, self.n_cells, self.n_cells
             )
        else:
             check_collisions_v1_numba(self.positions, self.velocities, self.radii, self.masses)
        print("Compilation complete.")

    def move(self, step: int):
        t0 = time.perf_counter()
        
        # 1. Move
        move_particles_kernel(self.positions, self.velocities, self.radii, self.config.BOX_SIZE, self.config.DT)
        
        # 2. Collide
        checks = 0
        if self.config.COLLISION_VERSION == 1:
            checks = check_collisions_v1_numba(self.positions, self.velocities, self.radii, self.masses)
        elif self.config.COLLISION_VERSION == 2:
            checks = check_collisions_v2_numba(self.positions, self.velocities, self.radii, self.masses)
        elif self.config.COLLISION_VERSION == 3:
            # Pass pre-allocated arrays
            checks = check_collisions_v3_numba(
                self.positions, self.velocities, self.radii, self.masses,
                self.grid_head, self.grid_next, self.cell_size, self.n_cells, self.n_cells
            )
            
        dt = time.perf_counter() - t0
        
        self.metrics['time_ms'] = dt * 1000
        self.metrics['checks'] = checks
        self.metrics['flops'] = checks * 15 + (self.config.NUM_PARTICLES * 10)
        self.metrics['memory_mb'] = self.process.memory_info().rss / (1024 * 1024) 
        self.total_elapsed_time += dt
        
        # 3. Log to file (Using open handle)
        self.perf_log_file.write(f"{step},{self.metrics['time_ms']:.5f},{self.metrics['memory_mb']:.2f},{checks},{self.metrics['flops']}\n")

        # 4. Save State
        if self.config.WRITE_STATE:
            if self.file_counter < self.config.FILES_LIMIT:
                self.file_counter += 1
                self.save_state(f'simulation_state_step_{step}.csv')
            elif self.file_counter == self.config.FILES_LIMIT:
                print(f"\nWarning: File limit reached ({self.config.FILES_LIMIT}). Stopping state dump.")
                self.file_counter += 1

        if step % 50 == 0:
             print(f"\rStep {step}: {self.metrics['time_ms']:.3f}ms, {checks} checks", end="")

    def update_animation(self, step, text_artist):
        self.move(step)
        self.particles.set_offsets(self.positions)
        text_artist.set_text(
            f"Step: {step}\nTime: {self.metrics['time_ms']:.2f} ms\n"
            f"Checks: {self.metrics['checks']:,}\nNumba Optimized"
        )
        return self.particles, text_artist

    def run_simulation(self, num_steps: int, animate: bool = False):
        if animate:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(0, self.config.BOX_SIZE)
            ax.set_ylim(0, self.config.BOX_SIZE)
            perf_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', family='monospace', bbox=dict(facecolor='white', alpha=0.8))
            self.particles = ax.scatter(self.positions[:, 0], self.positions[:, 1], s=(self.radii * self.config.SCATTER_SCALE)**2, alpha=0.6)
            
            ani = animation.FuncAnimation(fig, self.update_animation, fargs=(perf_text,), frames=num_steps, interval=1, blit=True)
            plt.show()
        else:
            for step in range(num_steps):
                self.move(step)
            print(f'\nTotal time: {self.total_elapsed_time:.4f}s')
        
        # Clean up file at end of run
        if not self.perf_log_file.closed:
            self.perf_log_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--animate', type=int, required=True)
    args = parser.parse_args()
    
    config = SimulationConfig(NUM_PARTICLES=1000, COLLISION_VERSION=args.version, WRITE_STATE=True)
    sim = ParticleSimulation(config)
    sim.initialize_particles()
    sim.run_simulation(num_steps=10, animate=args.animate == 1)

if __name__ == '__main__':
    main()
