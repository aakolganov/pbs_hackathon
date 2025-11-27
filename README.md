# HPC hackathon
🚀 Particle Collision Performance Challenge

## Overview
Welcome to the Particle Collision Performance Challenge! Your team's mission is to optimize a 2D 
particle simulation system that models the behavior of particles undergoing elastic collisions. 
The current implementation, while functional, has significant performance bottlenecks that need 
to be addressed.

## The Challenge
The simulation represents particles as circles with different radii moving in a 2D box. Each 
particle has:
- A position (x, y)
- A velocity vector
- A radius
- A mass (derived from radius)

Your task is to improve the performance of the collision detection and resolution system while 
maintaining physical accuracy. The current naive implementation checks every possible pair of 
particles for collisions, resulting in O(N²) complexity, where N is the total number of particles.

## Starting Point
- Base implementation uses NumPy for array operations
- Current collision detection is brute force (checking all pairs)
- Simulation parameters are configurable (number of particles, box size, etc.)
- Performance is measured by elapsed time per simulation step

## Objectives
1. **Primary Goal**: Reduce the average time per simulation step while maintaining accuracy
2. **Secondary Goals**:
    - Profile the script with, e.g., cProfile and memory-profiler
    - Define if the script is a compute-bound or memory-bound
    - Implement better collision detection algorithms
    - Optimize the collision resolution function
    - Optimize memory usage
    - Utilize parallel processing where applicable

## Rules
1. The physical behavior of particles must remain accurate
2. All optimizations must be documented and explained

## Evaluation Criteria
Teams will be judged on:
1. Performance improvement (50%)
2. Code quality and documentation (25%)
3. Innovation in approach (15%)
4. Presentation of the solution (10%)

## Technical Details
### Current Performance Bottlenecks:
``` python
def check_collisions_v1(self) -> None:
    for i in range(self.config.NUM_PARTICLES):
        for j in range(i + 1, self.config.NUM_PARTICLES):
            # O(n²) complexity
```
### Possible Optimization Approaches:
- Spatial partitioning
- Vectorized operations
- Multi-threading
- GPU acceleration
- Collision prediction

## Getting Started
1. Clone the repository
2. Install the requirements
    ``` bash
    pip install -r requirements.txt
    ```
2. Run the baseline simulation:
    ``` bash
    python main.py --version 1 --animate 0
    ```
3. Measure initial performance with 1000 particles

## Submission Requirements
1. Optimized implementation
2. Documentation of approaches tried
3. Performance benchmarks
4. Presentation of your solution

## Hints
- Consider using spatial data structures
- Look into NumPy's vectorized operations
- Think about ways to reduce unnecessary collision checks
- Consider parallel processing possibilities

May the fastest code win! 🏆
