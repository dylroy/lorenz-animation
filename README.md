**ACTIVE WORK-IN-PROGRESS**
# lorenz_animation
Animation structure for the 1963 Lorenz system of ordinary differential equations, using a Runge-Kutta integrator within a fourth-order scheme.

Within the Lorenz system of ODEs, there are three constants which can be changed; $\sigma$, $\rho$, and $\beta$. Doing so greatly varies the trajectory of the position vector as it is integrated forward through time. For some combinations of these three constants, the system behaves deterministically, or it at least appears to do so. For other combinations of these three constants, the system displays the wildly chaotic nature for which it is well known, albiet it in different ways depending on the coefficient selection. Method to the madness. The system is described below (note the locations of the coefficients in the system of equations):

$$ \frac{dx}{dt} = \sigma \left( y - x \right), \\\\
\frac{dy}{dt} = x \left( \rho - z \right) - y, \\\\
\frac{dz}{dt} = xy - \beta z$$

Integration function created with lots of help from 
Steve Brunton's YouTube channel, incorporation of functions into
an animation scheme with changeable Lorenz parameters was done 
by Dylan Roy.
