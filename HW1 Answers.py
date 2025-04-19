import abc
from typing import Callable
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import functools
import cvxpy as cp
import math
from jax import grad, jit, vmap

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
plt.clf
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

    # Get vals from state
    theta = state[2]
    v = state[3]

    A = jnp.array([
        [0, 0, -v *jnp.sin(theta), jnp.cos(theta)], #wrt xdot
        [0, 0, v *jnp.cos(theta), jnp.sin(theta)], #wrt ydot
        [0, 0, 0, 0], #wrt omega
        [0, 0, 0, 0] #wrt accel
    ])  #cols wrp x, y, theta, v


    B = jnp.array([
        [0, 0], #wrt xdor
        [0, 0], #wrt ydot
        [1, 0], #wrt omega
        [0, 1] #wrt accel
    ]) #cols wrt omega, accel

    C = jnp.array([0,0,0,0])

    return A, B, C

# b # Evaluate linearized dynamics (analytic)

x0 = np.transpose([0, 0, math.pi/4, 2])
u0 = np.transpose([0.1,1.])
time = .1
#print(linearize_unicycle_continuous_time_analytic(x0, u0, time))

# c # Linearize dynamics using JAX autodiff

def linearize_autodiff(function_name, state, control, time):
    '''
    Linearizes the any dynamics using jax autodiff.
    Inputs:
        function_name: name of function to be linearized. Takes state, control, and time as inputs.
        state     : A jax.numpy array of size (n,); the state to linearize about
        control   : A jax.numpy array of size (m,); the control to linearize about
        time      : A real scalar; the time to linearize about

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
        

    # test code:
state = jnp.array([0.0, 0.0, jnp.pi/4, 2.])
control = jnp.array([0.1, 1.])
time = 0.

A_autodiff, B_autodiff, C_autodiff = linearize_autodiff(continuous_dynamics, state, control, time)
A_analytic, B_analytic, C_analytic = linearize_unicycle_continuous_time_analytic(state, control, time)

print('A matrices match:', jnp.allclose(A_autodiff, A_analytic))
print('B matrices match:', jnp.allclose(B_autodiff, B_analytic))
print('C matrices match:', jnp.allclose(C_autodiff, C_analytic))

# d # Linearize discrete-time dynamics 
state = jnp.array([0.0, 0.0, jnp.pi/4, 2.])
control = jnp.array([0.1, 1.])
time = 0.1

ARK_autodiff, BRK_autodiff, CRK_autodiff = linearize_autodiff(discrete_dynamics_rk, state, control, time)
ARK_analytic, BRK_analytic, CRK_analytic = linearize_unicycle_continuous_time_analytic(state, control, time)

AE_autodiff, BE_autodiff, CE_autodiff = linearize_autodiff(discrete_dynamics_euler, state, control, time)
AE_analytic, BE_analytic, CE_analytic = linearize_unicycle_continuous_time_analytic(state, control, time)

print('A matrices match:', jnp.allclose(ARK_autodiff, AE_analytic))
print('B matrices match:', jnp.allclose(BRK_autodiff, BE_analytic))
print('C matrices match:', jnp.allclose(CRK_autodiff, CE_analytic))

# e # 

key = jax.random.PRNGKey(42)  # Set a fixed seed
n_samples = 1000
state_dim = 4  # 4-dimensional state
ctrl_dim = 2  # 2-dimensional control

time = 0.0
random_states = jax.random.normal(key, shape=(n_samples, state_dim))
random_controls = jax.random.normal(key, shape=(n_samples, ctrl_dim))

trajs = jax.vmap(lambda init, u: linearize_autodiff(discrete_dynamics_rk, init, u, 0.1))(random_states, random_controls)

# plot the trajectories
# Better way to visualize???
plt.clf()

plt.plot(trajs[0][:, 0], trajs[0][:, 1], label=f"Trajectory 0", linestyle='dotted')
#plt.plot(trajs[1][:, 0], trajs[1][:, 1], label=f"Trajectory 1", linestyle='solid')
#plt.plot(trajs[2][:, 0], trajs[2][:, 1], label=f"Trajectory 2", linestyle='dashdot')

#plt.plot(trajs[0, :, 0], trajs[1, :, 1], label=f"dt = {0.1} RK", linestyle='dotted')
plt.legend()
plt.grid(alpha=0.4)
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
#plt.show() # UNCOMMENT FOR GRAPH
# print(trajs) # UNCOMMENT FOR VALS

## COME BACK TO 2E GRAPHING ## 

## P3 ## Unconstrained Optimization

# a # Gradient descent on unconstrained optimization problem

def f(x):
    return (x + 2)**2 + 5*jnp.tanh(x)

plt.clf
args = np.arange(-6,4,0.01)
plt.figure(figsize=(8,6))
plt.plot(args, f(args))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Objective function')
plt.grid(alpha=0.3)
plt.show()

def minimize_with_gradient_descent(func, initial_guess, step_size, convergence_tol=1e-8):
    '''
    Minimizes a scalar function of a single variable.
    Inputs:
        func              : name of function to be optimized. Takes initial_guess as input.
        initial_guess     : a real number
        convergence_tol   : convergence tolerace; when current and next guesses of of optimal x are closer
                            together than this, algorithm terminates and returns current estimate of optimal x

    Outputs:
        cur_x : current best estimate of x which minimizes f(x)
    '''

    next_x = cur_x = initial_guess  #init 
    current_tol = convergence_tol
    deriv_func = grad(func)
    while current_tol >= convergence_tol:
        
        next_x = cur_x - step_size * deriv_func(cur_x)
        current_tol = abs(cur_x - next_x) # calculate new tol
        cur_x = next_x # update curr

    return cur_x

x_opt = minimize_with_gradient_descent(f, 5.0, 0.1)


# output and plot:
print('optimal x:', x_opt)
print('optimal value of f(x):', f(x_opt))

args = np.arange(-6,4,0.01)
plt.figure(figsize=(8,6))
plt.plot(args, f(args), label='f(x)')
plt.scatter(x_opt, f(x_opt), zorder=2, color='red', label='optimal point')
plt.title('x_opt = {:.4f}, f(x_opt) = {:.4f}'.format(x_opt, f(x_opt)))
plt.grid(alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
#plt.show()

# b # Applying log-barrier for solving constrained optimization problems

# fill out g(x) so that the statement g(x) < 0 is equivalent to the statement x > 1
def g(x):
    return 1 - x

def phi(f, x, g, t):
    '''
    Computes phi(x).
    Inputs:
        f  : name of f(x) function; takes x as input
        x  : variable to be optimized
        g  : constraint function; we want to ensure g(x) <= 0
        t  : log-barrier weighting parameter

    Outputs:
        phi(x)
    '''
    phi_x = f(x) - t*jnp.log(-g(x))

    return phi_x

x_upper = 4
dx = 0.01
f_x_domain = np.arange(-6, x_upper, dx)
phi_x_domain = np.arange(1.00001, x_upper, dx)

plt.figure(figsize=(8,6))
plt.plot(f_x_domain, f(f_x_domain), label='f(x)')
plt.plot(phi_x_domain, phi(f, phi_x_domain, g, 5), label='phi(x), t = 5')
plt.plot(phi_x_domain, phi(f, phi_x_domain, g, 2), label='phi(x), t = 2')
plt.plot(phi_x_domain, phi(f, phi_x_domain, g, 0.5), label='phi(x), t = 0.5')
plt.vlines(1, -10, 40, linestyles='dashed', label='x = 1', color='black')
plt.xlabel('x')
plt.grid(alpha=0.3)
plt.ylabel('f(x), phi(x)')
plt.title('f(x) and phi(x) vs x')
plt.legend(loc='upper left')
# plt.ylim(-10, 40)
plt.show()

# c # minimize_with_gradient descent
# ========================== hint: lambdas ==============================

def hint_func(arg1, arg2, arg3):
    return arg1 + 2 * arg2 + arg3

def hint_func_caller(func, arg):
    return func(arg)

foo = 42
bar = 100

lambda x: hint_func(foo, x, bar) # this essentially defines a function of x only, that calls hint_func(x, 42, 100)

# hint_func_caller expects a function that takes only one argument. But we can use a lambda expression to
# "prepopulate" all but one argument of hint_func, "turning it into" an argument of one function:

hint_func_caller(lambda x: hint_func(foo, x, bar), 5) # this will work and give 152

# ^^^^^^^^^^^^^^^^^^^^^^^^^^ hint: lambdas ^^^^^^^^^^^^^^^^^^^^^^^^^^

# ========================== hint: functools.partial ==============================

new_func = functools.partial(hint_func, foo, arg3=bar) # this is equivalent to lambda x: hint_func(foo, x, bar)
new_func(5) # this will give 152

# OR

new_func = functools.partial(hint_func, arg1=foo, arg3=bar) # this is equivalent to lambda x: hint_func(foo, x, bar)
new_func(arg2=5) # this will give 152

# ^^^^^^^^^^^^^^^^^^^^^^^^^^ hint: functools.partial ^^^^^^^^^^^^^^^^^^^^^^^^^^




# add you code here


