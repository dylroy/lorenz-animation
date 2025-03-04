"""
Runge-Kutta Integrator:

Using the RK4 integrator to integrate the Lorenz 1963 
system of ordinary differential equations

Integration function created with lots of help from 
Steve Brunton's YouTube channel, incorporation of functions into
an animation scheme with changeable Lorenz parameters was done 
by Dylan Roy.
"""

#Import required modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

#Lorenz's parameters for chaotic system

#Integrator function
def rk4singlestep(fun, dt, Xk, tk, sigma, rho, beta):
    """
    **Inputs: function, time step, starting x value, starting t value
    **Output: x out (the single step integration of Lorenz function at
    time tk and at xk point in space)
    """
    f1=fun(Xk,
           tk,
           sigma,rho,beta)
    f2=fun(Xk+(dt/2)*f1,
           tk+(dt/2),
           sigma,rho,beta)
    f3=fun(Xk+(dt/2)*f2,
           tk+(dt/2),
           sigma,rho,beta)
    f4=fun(Xk+dt*f3,tk+dt,
           sigma,rho,beta)
    Xout= Xk+(dt/6)*(f1*2+f2*2+f3*2+f4*2)
    return Xout

#Lorenz function
def lorenz(X, t, sigma, rho, beta):
    """
    System of Ordinary Differential Equations (ODEs). Used as input to the rk4singlestep
    function which integrates using the Runge-Kappe fourth-order scheme

    **Inputs: starting value in space (x), starting value in time (t),
    sigma (constant; Prandtl number), rho (constant; Rayleigh number),
    beta (constant; related to the aspect ratio of the convection rolls)
    **Outputs: vector of the rates of each spatial dimension (x,y,z) with
    respect to time
    """
    #X is a three-dimensional state vector where: X[1]=x, X[2]=y, X[3]=z
    dX=[
        sigma*(X[1]-X[0]),
        X[0]*(rho-X[2])-X[1],
        X[0]*X[1]-beta*X[2]
    ]
    return np.array(dX)

#Initial conditions
Xk = [-8, 8, 27]
tk = 0
Xk2 = [-7.9, 8.1, 27.1]

#Compute tranjectory
def iter(sigma, rho, beta, Xk1, Xk2):
    dt = 0.01 #Change dt here
    T = 10 #Change time here
    num_time_pts = int(T/dt)
    t = np.linspace(0, T, num_time_pts)

    X1 = np.zeros((3, num_time_pts))
    X2 = np.zeros((3, num_time_pts))

    X1[:, 0] = Xk1
    X2[:, 0] = Xk2
    Xin1=Xk1
    Xin2=Xk2
    for i in range(num_time_pts-1):
        Xout1 = rk4singlestep(lorenz, dt, Xin1, t[i], sigma, rho, beta)
        Xout2 = rk4singlestep(lorenz, dt, Xin2, t[i], sigma, rho, beta)
        X1[:, i+1] = Xout1
        X2[:, i+1] = Xout2
        Xin1=Xout1
        Xin2=Xout2
    return X1, X2

#Plot Figure
fig2 = plt.figure(1, dpi=300)
ani_ax = fig2.add_subplot(projection='3d')

def update(frame):
    ani_ax.clear()
    if frame <= 240:
        Xk = [-8, 8, 27] #Primary starting location vector
        Xk2 = [-7.9, 8.1, 27.1] #Secondary starting location vector
        fsigma = 0 + round(frame/30, 2)
        frho = 28
        fbeta = round(8/3, 2)
        X1, X2 = iter(fsigma, frho, fbeta, Xk, Xk2)
        ani_return = ani_ax.plot(X1[0, :], X1[1, :], X1[2, :], 'b', linewidth = 0.5)
        ani_return = ani_ax.plot(X2[0, :], X2[1, :], X2[2, :], 'r', linewidth = 0.5)
    elif 240 < frame <= 480:
        Xk = [-8, 8, 27] #Primary starting location vector
        Xk2 = [-7.9, 8.1, 27.1] #Secondary starting location vector
        fsigma = 10
        frho = 0 + round((frame-240)/5, 2)
        fbeta = round(8/3, 2)
        X1, X2 = iter(fsigma, frho, fbeta, Xk, Xk2)
        ani_return = ani_ax.plot(X1[0, :], X1[1, :], X1[2, :], 'b', linewidth = 0.5)
        ani_return = ani_ax.plot(X2[0, :], X2[1, :], X2[2, :], 'r', linewidth = 0.5)
    elif 480 < frame <= 720:
        Xk = [-8, 8, 27] #Primary starting location vector
        Xk2 = [-7.9, 8.1, 27.1] #Secondary starting location vector
        fsigma = 10
        frho = 28
        fbeta = 0 + round((frame-480)/80, 2)
        X1, X2 = iter(fsigma, frho, fbeta, Xk, Xk2)
        ani_return = ani_ax.plot(X1[0, :], X1[1, :], X1[2, :], 'b', linewidth = 0.5)
        ani_return = ani_ax.plot(X2[0, :], X2[1, :], X2[2, :], 'r', linewidth = 0.5)
    ani_ax.set_title('Lorenz Attractor', fontname='serif')
    ani_ax.legend([rf"$\sigma = ${fsigma}" + "\n" + rf"$\rho =${frho}" + "\n" + rf"$\beta =${fbeta}"], loc='upper right')
    ani_ax.set_xlim(-35,35)
    ani_ax.set_ylim(-35,35)
    ani_ax.set_zlim(0,50)
    ani_ax.set_xlabel(rf"$x_{1}(f)$")
    ani_ax.set_ylabel(rf"$x_{2}(f)$")
    ani_ax.set_zlabel(rf"$x_{3}(f)$")
    return ani_return

ani = animation.FuncAnimation(fig=fig2, func=update, frames=720, interval=1, repeat=True)
ani.save("LorenzAttractor7.mp4", writer="ffmpeg", fps=15)

plt.show() 