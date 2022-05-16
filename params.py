import dolfin

# ----------------------
# (I) General parameters
# ----------------------

# (I.1) Material parameters
# -------------------------
L_m = 335e3                 # Latent heat of melting (water--ice) [J/kg]
rho_s = 917.                # Density of ice [kg*m**(-3)]
rho_l = 1000.               # Density of water [kg*m**(-3)]
rho = rho_l
c_s = 2116.                # Specific heat of ice
c_l = 4182.                # Specific heat of water
c_m = (c_l + c_s)/2        # Specific heat of the mushy region
k_s = 2.26                  # Thermal conductivity of ice [W/m/K]
k_l = 0.6                  # Thermal conductivity of water [W/m/K]
mu_s = 1e6                  # Dynamic viscosity of ice [Pa*s]
mu_l = 1.51e-3                  # Dynamic viscosity of water [Pa*s]
nu_s = mu_s/rho_s           # Kinematic viscosity of ice [m**2/s]
nu_l = mu_l/rho_l           # Kinematic viscosity of water [m**2/s]
alpha_s = k_s/(rho_s*c_s)  # Heat diffusivity of ice
alpha_l = k_l/(rho_l*c_l)  # Heat diffusivity of water

# (I.2) Physical parameters
# -------------------------
g = 9.81                    # Gravitational acceleration
theta_m = 273.              # Melting temperature [K]

# (I.3) Geometric parameters
# --------------------------

# No-slip boundary condition in 2d:
noslip = dolfin.Constant((0., 0.))

# Unit vector, upward direction:
ez = dolfin.Constant((0., 1.))
# ====================================

# --------------------------------
# (II) Problem specific parameters
# --------------------------------

# (II.1) Hagen-Poiseuille benchmark
# ---------------------------------
# Channel geometry (2d rectangular channel (0,l_x) x (-l_z,l_z)):
l_x = 1.0; l_z = .014       # Length and half-width of the channel
# Channel geometry (h-axisym: (0,l_x) x (0,R), v-axisym: (0,R) x (0,l_z)):
R = 0.006                   # Pipe radius

nx = 100; nz = 100; nr = 20 # Mesh cells in x, z and r direction

U_in = 10.0                 # Peak inflow velocity

# (II.2) Stokes drag benchmark
# ----------------------------

# (II.3) Stefan benchmark
# -----------------------

R1 = 0.0; R2 = 1.

# Heat source power constant (Q_d(t) = Q_0*2^(d-1)|S_d(1)|*k_l*(alpha_l*t)^(d/2-1)):
Q_0 = 50.

# Initial temperature:
theta_i = 263.

# (II.4) COMSOL benchmark
# -----------------------
# Sphere initial position and radius:
sphere_position = dolfin.Point(0.0, 0.0); sphere_radius = 0.001
sphere_rho = 2.9e3; sphere_mass = 4./3.*dolfin.pi*sphere_radius**3*sphere_rho


# ================================

# (II.1) Geometric
# ----------------
L = 0.1                     # Domain length
H = 0.1                     # Domain height

meshres = [1000,100]           # Rectangular mesh resolution
#====================================
