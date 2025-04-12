import abc
from typing import Callable
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import functools
import cvxpy as cp
import math


## PROBLEM 1 a ##

class Dynamics(metaclass=abc.ABCMeta):
    dynamics_func: Callable
    state_dim: int
    control_dim: int

    def __init__(self, dynamics_func, state_dim, control_dim):
        self.dynamics_func = dynamics_func
        self.state_dim = state_dim
        self.control_dim = control_dim

    def __call__(self, state, control, time=0):
        return self.dynamics_func(state, control, time)
    
def dynamic_unicycle_ode(state, control, time):
    x, y, theta, v = state
    omega = control[0]
    a = control[1]
    dxdt = v*jnp.cos(theta)
    dydt = v*jnp.sin(theta)
    dthetadt = omega
    dvdt = a

    return jnp.array([dxdt, dydt, dthetadt, dvdt])

state_dim = 4
control_dim = 2
continuous_dynamics = Dynamics(dynamic_unicycle_ode, state_dim, control_dim)


# b # Obtaining discrete-time dynamics

def euler_integrate(dynamics, dt):
    # zero-order hold
    def integrator(x, u, t):
        dx = dynamics(x, u, t)
        return x + dt * dx
    return integrator

def runge_kutta_integrator(dynamics, dt=0.1):
    # zero-order hold
    def integrator(x, u, t):
        k1 = dynamics(x, u, t)
        k2 = dynamics(x + 0.5 * dt * k1, u, t + 0.5 * dt)
        k3 = dynamics(x + 0.5 * dt * k2, u, t + 0.5 * dt)
        k4 = dynamics(x + dt * k3, u, t + dt)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return integrator

# c #

def simulate(dynamics, initial_state, controls, dt):
    state = initial_state
    trajectory = [state]
    time = 0.0

    for u in controls:
        state = dynamics(state, u, time)
        trajectory.append(state)
        time += dt

    return jnp.stack(trajectory)


# code to loop over the different integrators and step sizes
# and plot the corresponding trajectories

initial_state = jnp.array([0.0, 0.0, 0.0, 0.0])
control = jnp.array([2.0, 1.0])  # constant control over the 5 second duration.
duration = 5.0
dts = [0.01, 0.5]

for dt in dts:
    num_steps = int(duration / dt)
    controls = [control] * num_steps
    
    # construct the discrete dynamics for given timestep
    discrete_dynamics_euler = Dynamics(
        euler_integrate(continuous_dynamics, dt), state_dim, control_dim
    )
    discrete_dynamics_rk = Dynamics(
        runge_kutta_integrator(continuous_dynamics, dt), state_dim, control_dim
    )
    
    # simulate dynamics
    xs_euler = simulate(discrete_dynamics_euler, initial_state, controls, dt)
    xs_rk = simulate(discrete_dynamics_rk, initial_state, controls, dt)

    # plot the trajectories
    plt.plot(xs_euler[:, 0], xs_euler[:, 1], label=f"dt = {dt} Euler",linestyle='--')
    plt.plot(xs_rk[:, 0], xs_rk[:, 1], label=f"dt = {dt} RK",linestyle='dotted')
    plt.legend()

#plt.show()     # UNCOMMENT FOR GRAPH
plt.grid(alpha=0.4)
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")

# d #

def foo(x, y, z):
    return x + y + z

N = 1000
x = jnp.array(np.random.randn(N))
y = jnp.array(np.random.randn(N))
z = jnp.array(np.random.randn(N))

xs = jnp.array(np.random.randn(N, N))
ys = jnp.array(np.random.randn(N, N))
zs = jnp.array(np.random.randn(N, N))

foo(x, y, z)  # non-vectorized version
# vectorized version for all inputs, 0 is the batch dimension for all inputs
jax.vmap(foo, in_axes=[0, 0, 0])(xs, ys, zs)  

# x not batched, but ys and zs are with 0 as the batch dimension
jax.vmap(foo, in_axes=[None, 0, 0])(x, ys, zs)  

# y not batched, but xs and zs are with 0 as the batch dimension
jax.vmap(foo, in_axes=[0, None, 0])(xs, y, zs)  

# z not batched, but xs and ys are with 0 as the batch dimension
jax.vmap(foo, in_axes=[0, 0, None])(xs, ys, z)  

# x and y not batched, but zs is with 0 as the batch dimension
jax.vmap(foo, in_axes=[None, None, 0])(x, y, zs)  

# vectorized version for all inputs, batch dimension for xs is 1, 
# while 0 is the batch dimension for yx and zs
jax.vmap(foo, in_axes=[1, 0, 0])(xs, ys, zs)  

state_dim = continuous_dynamics.state_dim
control_dim = continuous_dynamics.control_dim
N = 1000 # num of trajectories
n_time_steps = 50
initial_states = jnp.array(np.random.randn(N, state_dim))
controls = jnp.array(np.random.randn(N, n_time_steps, control_dim))

## Editing ##
trajs = jax.vmap(lambda init, u: simulate(runge_kutta_integrator(continuous_dynamics, 0.1), init, u, 0.1))(initial_states, controls)

# plot the trajectories
# Better way to visualize???
plt.plot(trajs[:, 0], trajs[:, 1], label=f"dt = {0.1} RK",linestyle='dotted')
plt.legend()
plt.grid(alpha=0.4)
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
#plt.show() # UNCOMMENT FOR GRAPH
# print(trajs) # UNCOMMENT FOR VALS

# e # 
# # without jitting
# # timeit jax.vmap(simulate, in_axes=[None, 0, 0, None])(discrete_dynamics_rk, initial_states, controls, jnp.array(dt)).block_until_ready()

# # method 1: directly apply jax.jit over the jax.vmap function
# # need to provide the static_argnums argument to the first argument since that is a function input and not an array input
# sim_jit = jax.jit(jax.vmap(simulate, in_axes=[None, 0, 0, None]), static_argnums=0)
        
# # time the run
# %timeit sim_jit(discrete_dynamics_rk, initial_states, controls, jnp.array(dt)).block_until_ready()

# # method 2: apply jax.jit over the simulate function and then apply jax.vmap
# sim_jit = jax.jit(simulate, static_argnums=0)
# sim_jit_vmap = jax.vmap(sim_jit, in_axes=[None, 0, 0, None])
# %timeit sim_jit_vmap(discrete_dynamics_rk, initial_states, controls, jnp.array(dt)).block_until_ready()

# # Method 3: apply jax.jit over the simulate function during function construction and then apply jax.vmap
# @functools.partial(jax.jit, static_argnames=("dynamics"))
# def simulate(dynamics, initial_state, controls, dt):
#     xs = [initial_state]
#     time = 0
#     for u in controls:
#         xs.append(dynamics(xs[-1], u, time))
#         time += dt
#     return jnp.stack(xs)

# sim_jit_vmap = jax.vmap(simulate, in_axes=[None, 0, 0, None])
# %timeit sim_jit_vmap(discrete_dynamics_rk, initial_states, controls, jnp.array(dt)).block_until_ready()

    # a # Linearize dynamics analytically
def linearize_unicycle_continuous_time_analytic(state, control, time):
    '''
    Linearizes the continuous time dynamics of the dynamic unicyle using analytic expression
    Inputs:
        state     : A jax.numpy array of size (n,)
        control   : A jax.numpy array of size (m,)
        time      : A real scalar

    Outputs:
        A : A jax.numpy array of size (n,n)
        B : A jax.numpy array of size (n,m)
        C : A jax.numpy array of size (n,1)
    '''
    func = lambda state, control, time: dynamic_unicycle_ode(state, control, time)
    A = jax.jacobian(func, argnums = 0)(state, control, time)
    B = jax.jacobian(func, argnums = 1)(state, control, time)
    C = jax.jacobian(func, argnums = 2)(state, control, time) #time does not affect, should be 0


    print(A,B,C)
    return A, B, C

# b # Evaluate linearized dynamics (analytic)

x0 = np.transpose([0, 0, math.pi/4, 2])
u0 = np.transpose([0.1,1.])
time = .1
print(linearize_unicycle_continuous_time_analytic(x0, u0, time))