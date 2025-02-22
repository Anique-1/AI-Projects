import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from datetime import datetime, timedelta

class SpaceObject:
    def __init__(self, position, velocity, size, mass, name):
        # Ensure position and velocity are float arrays
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.size = float(size)
        self.mass = float(mass)
        self.name = name
        self.trajectory_history = [self.position.copy()]
        
    def update_position(self, dt):
        # Convert dt to float and perform update
        dt = float(dt)
        self.position = self.position + self.velocity * dt
        self.trajectory_history.append(self.position.copy())

class DebrisTracker:
    def __init__(self):
        self.objects = []
        self.collision_threshold = 1.0  # km
        
    def add_object(self, space_object):
        self.objects.append(space_object)
    
    def calculate_collision_risk(self, obj1, obj2):
        distance = np.linalg.norm(obj1.position - obj2.position)
        relative_velocity = np.linalg.norm(obj1.velocity - obj2.velocity)
        
        # Ensure float division
        risk = 1.0 / (distance * relative_velocity + 1e-6)
        return float(risk)
    
    def find_collision_risks(self):
        risks = []
        for i, obj1 in enumerate(self.objects):
            for j, obj2 in enumerate(self.objects[i+1:], i+1):
                risk = self.calculate_collision_risk(obj1, obj2)
                if risk > 0.1:  # Risk threshold
                    risks.append((obj1, obj2, risk))
        return risks
    
    def plan_avoidance_maneuver(self, obj1, obj2):
        relative_position = obj2.position - obj1.position
        avoid_direction = np.cross(relative_position, obj1.velocity)
        norm = np.linalg.norm(avoid_direction)
        
        # Avoid division by zero
        if norm > 1e-10:
            avoid_direction = avoid_direction / norm
        else:
            avoid_direction = np.array([1.0, 0.0, 0.0])
        
        risk = self.calculate_collision_risk(obj1, obj2)
        maneuver_magnitude = float(risk * 0.1)  # km/s
        
        return avoid_direction * maneuver_magnitude

class OrbitPropagator:
    def __init__(self):
        self.G = 6.67430e-11  # gravitational constant
        self.M = 5.972e24     # Earth mass
        
    def acceleration(self, state, t):
        x, y, z, vx, vy, vz = [float(val) for val in state]
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Gravitational acceleration
        ax = -self.G * self.M * x / r**3
        ay = -self.G * self.M * y / r**3
        az = -self.G * self.M * z / r**3
        
        return [vx, vy, vz, ax, ay, az]
    
    def propagate(self, initial_state, t):
        initial_state = np.array(initial_state, dtype=np.float64)
        t = np.array(t, dtype=np.float64)
        return odeint(self.acceleration, initial_state, t)

def create_demo_scenario():
    # Create demo space objects with explicit float values
    satellite = SpaceObject(
        position=np.array([42164.0, 0.0, 0.0]),  # GEO orbit
        velocity=np.array([0.0, 3.075, 0.0]),
        size=10.0,
        mass=1000.0,
        name="Active Satellite"
    )
    
    debris = SpaceObject(
        position=np.array([42164.0, 100.0, 50.0]),
        velocity=np.array([0.1, 3.0, 0.1]),
        size=0.1,
        mass=1.0,
        name="Debris"
    )
    
    return satellite, debris

def main():
    st.title("Space Debris Tracking and Collision Avoidance Simulation")
    
    # Initialize simulation
    tracker = DebrisTracker()
    propagator = OrbitPropagator()
    
    # Create demo scenario
    satellite, debris = create_demo_scenario()
    tracker.add_object(satellite)
    tracker.add_object(debris)
    
    # Simulation controls
    st.sidebar.header("Simulation Controls")
    time_step = st.sidebar.slider("Time Step (hours)", 0.1, 24.0, 1.0)
    num_steps = st.sidebar.slider("Number of Steps", 10, 1000, 100)
    
    # Run simulation
    if st.button("Run Simulation"):
        progress_bar = st.progress(0)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for step in range(num_steps):
            # Update positions
            for obj in tracker.objects:
                obj.update_position(float(time_step * 3600))  # Convert hours to seconds
            
            # Check for collision risks
            risks = tracker.find_collision_risks()
            
            # Plot
            ax.clear()
            
            # Plot Earth
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = 6371.0 * np.outer(np.cos(u), np.sin(v))
            y = 6371.0 * np.outer(np.sin(u), np.sin(v))
            z = 6371.0 * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='b', alpha=0.1)
            
            # Plot objects and their trajectories
            for obj in tracker.objects:
                trajectory = np.array(obj.trajectory_history, dtype=np.float64)
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r-', alpha=0.5)
                ax.scatter(obj.position[0], obj.position[1], obj.position[2], 
                          label=obj.name, s=50)
            
            # Plot collision risks
            for obj1, obj2, risk in risks:
                mid_point = (obj1.position + obj2.position) / 2.0
                ax.text(mid_point[0], mid_point[1], mid_point[2], 
                       f'Risk: {risk:.2f}', color='red')
            
            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
            ax.set_zlabel('Z (km)')
            ax.legend()
            
            # Update progress
            progress_bar.progress(float(step + 1) / float(num_steps))
            
            # Display plot
            st.pyplot(fig)
            plt.close()
            
            # Display collision risks
            if risks:
                st.warning("Collision Risks Detected!")
                for obj1, obj2, risk in risks:
                    st.write(f"Risk between {obj1.name} and {obj2.name}: {risk:.2f}")
                    maneuver = tracker.plan_avoidance_maneuver(obj1, obj2)
                    st.write(f"Suggested avoidance maneuver: {maneuver}")

if __name__ == "__main__":
    main()