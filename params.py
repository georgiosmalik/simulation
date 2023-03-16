import dolfin

# ----------------------
# (I) General parameters
# ----------------------

# (I.1) Material parameters
# -------------------------

L_m = 335e3                 # Latent heat of melting (water-ice) [J/kg]

rho = 1000.                 # Single-phase density [kg*m**(-3)]
rho_s = 1000.               # Density of ice [kg*m**(-3)]
rho_l = 1000.               # Density of water [kg*m**(-3)]
rho_m = (rho_s + rho_l)/2   # Mean density (of mushy region) [kg*m**(-3)]

c_l = 4182.                # Specific heat of water [J/kg/K]
c_s = 2116.                # Specific heat of ice [J/kg/K]
c_m = (c_l+c_s)/2        # Specific heat of the mushy region

k_l = 0.6                   # Thermal conductivity of water [W/m/K]
k_s = 2.26                  # Thermal conductivity of ice [W/m/K]

mu_s = 1e6                  # Dynamic viscosity of ice [Pa*s]
mu_l = 1e-3                  # Dynamic viscosity of water [Pa*s]

nu_s = mu_s/rho_s           # Kinematic viscosity of ice [m**2/s]
nu_l = mu_l/rho_l           # Kinematic viscosity of water [m**2/s]

alpha_s = k_s/(rho_s*c_s)   # Heat diffusivity of ice
alpha_l = k_l/(rho_l*c_l)  # Heat diffusivity of water

beta_s = 0.0               # Thermal expansivity of ice
beta_l = 2.5e-4            # Thermal expansivity of water

# (I.2) Physical constants
# ------------------------

g = 9.81                    # Gravitational acceleration (magnitude) [m/s**2]
theta_m = 273.15              # Melting temperature of ice [K]

# (I.3) Geometric constants
# -------------------------

# No-slip boundary condition in 2d:
noslip = dolfin.Constant((0., 0.))

# Unit vector, upward direction:
ez = dolfin.Constant((0., 1.))
# ====================================
