# Sem bych chtel nastrkat veskery kod, ktery se tyka formulace problemu ve frniuvsd

import dolfin
import ufl
import numpy as np

import sim.params as prm

# ---------------------
# (0) Global parameters
# ---------------------

# (0.1) Temporal discretization
# -----------------------------

# Time discretization parameter THETA (0.0 = forward Euler):
THETA = 0.0

# Default time step size:
DT = dolfin.Constant(1e-3)

# (0.2) Spatial discretization
# ----------------------------

# FE degree (for mechanical Taylor-Hood, for temperature DEGREE+1):
DEGREE = 1

# Define unit vector pointing upwards:
ez = dolfin.Constant((0.0, 1.0))

# =================

# ---------
# FE spaces
# ---------

# Standard elements:
def CGdeg_ele(mesh, deg = DEGREE):

    return dolfin.FiniteElement("CG", mesh.ufl_cell(), deg)

def DGdeg_ele(mesh, deg = DEGREE):

    return dolfin.FiniteElement("DG", mesh.ufl_cell(), deg)

# Vector elements:
def vecCGdeg_ele(mesh, deg = DEGREE):

    return dolfin.VectorElement("CG", mesh.ufl_cell(), deg + 1)

# Special elements:

# Define MINI element (works with dolfin-version 2019.1.0):
def MINI_ele(mesh, deg = DEGREE):

    V_ele = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), deg)
    
    Q_ele = dolfin.FiniteElement("Bubble", mesh.ufl_cell(), mesh.topology().dim() + 1)

    return dolfin.VectorElement(ufl.NodalEnrichedElement(V_ele, Q_ele))

# Define piecwise polynomial scalar continous FE space:
def CGdeg(mesh, deg = DEGREE):

    return dolfin.FunctionSpace(mesh, CGdeg_ele(mesh, deg))

# Define piecwise constant scalar discontinous FE space:
def DGdeg(mesh, deg = 0):

    return dolfin.FunctionSpace(mesh, DGdeg_ele(mesh, deg))

def TaylorHood(mesh, deg = DEGREE):

    return



# =========

# General variational problem:
class Problem:

    def __init__(self, geometry):

        # Define default time step size:
        self.dt = DT

        # Define measures, measure weight, and normal:
        self.dx = dolfin.Measure("dx", domain = geometry.mesh)
        self.ds = dolfin.Measure("ds", domain = geometry.mesh, subdomain_data = geometry.boundary)
        self.d_w = 1.0
        self.n = dolfin.FacetNormal(geometry.mesh)

        # Define spatial coordinate:
        self.x = dolfin.SpatialCoordinate(geometry.mesh)

        # Initialize FE space and functions:
        self.W = None; self.w = None; self.w_k = None; self.w_kminus1 = None

        # Initialize auxiliary variables from other problem forms:
        self.var = None; self.var_k = None; self.var_kminus1 = None

        # Initialize form:
        self.F = 0

        # Initialize boundary conditions:
        self.bcs_essential = []; self.bcs_natural = []

    def define_solver(self, THETA = THETA, stationary = False):

        # Build form:
        self.define_form(THETA, stationary)

        # Return linear formulation:
        if not THETA:

            problem = dolfin.LinearVariationalProblem(dolfin.lhs(self.F),
                                                      dolfin.rhs(self.F),
                                                      self.w,
                                                      bcs = self.bcs_essential)

            self.solver = dolfin.LinearVariationalSolver(problem)

            return

        # Return non-linear formulation:
        else:

            # Define jacobian of the system:
            J = dolfin.derivative(self.F, self.w)

            problem = dolfin.NonlinearVariationalProblem(self.F, self.w, self.bcs_essential, J)

            self.solver = dolfin.NonlinearVariationalSolver(problem)

            # Define solver parameters:
            solver_prm = self.solver.parameters
            solver_prm['nonlinear_solver'] = 'newton'
            solver_prm['newton_solver']['linear_solver'] = 'mumps'
            solver_prm['newton_solver']['lu_solver']['report'] = True
            #solver_prm['newton_solver']['lu_solver']['same_nonzero_pattern']=True
            solver_prm['newton_solver']['absolute_tolerance'] = 1E-8
            solver_prm['newton_solver']['relative_tolerance'] = 1E-8
            solver_prm['newton_solver']['maximum_iterations'] = 15 # DBG
            solver_prm['newton_solver']['report'] = True
            #solver_prm['newton_solver']['error_on_nonconvergence'] = False

            return

    # Set initial condition (from expression, or constant):
    def set_initial(self, w_initial):

        # Use theta_initial expression to interpolate the function:
        self.w_k.assign(dolfin.interpolate(w_initial, self.W))

        # Assign the initial value to function for nonlinear solvers:
        self.w.assign(self.w_k)

    def assign_next_step(self):

        self.w_kminus1.assign(self.w_k)

        self.w_k.assign(self.w)

        return

# (I) Mechanical problems
# -----------------------

# Build Navier-Stokes problem formulation in cartesian geometry: 
class NavierStokesCartesian(Problem):

    def __init__(self, geometry):

        # Initialize general superstructure (form, boundary conditions, functions, geometry):
        Problem.__init__(self, geometry)

        # (I) FE spaces
        # -------------

        # Elements and spaces for mechanical problem (Taylor-Hood):
        V_ele = vecCGdeg_ele(geometry.mesh, DEGREE+1)
        P_ele = CGdeg_ele(geometry.mesh, DEGREE)

        # Elements and spaces for mechanical problem (Bubble for velocity, works with dolfin-version 2019.1.0):
        # V_ele = MINI_ele(geometry.mesh, DEGREE)
        # P_ele = CGdeg_ele(geometry.mesh, DEGREE)

        # Define mixed element for velocity and pressure:
        W_ele = dolfin.MixedElement([V_ele, P_ele])

        # Finite space for mechanical problem:
        self.W = dolfin.FunctionSpace(geometry.mesh, W_ele)

        # Separate finite spaces for velocities and pressures:
        self.V = dolfin.FunctionSpace(geometry.mesh, V_ele); self.P = dolfin.FunctionSpace(geometry.mesh, P_ele)

        # (II) Functions
        # --------------
        # Define general functions:
        self.w = dolfin.Function(self.W); self.v, self.p = dolfin.split(self.w)
        self.w_k = dolfin.Function(self.W); self.v_k, self.p_k = dolfin.split(self.w_k)
        self.w_kminus1 = dolfin.Function(self.W)

        # Define test and trial functions:
        self.v_, self.p_ = dolfin.TestFunctions(self.W); self._v, self._p = dolfin.TrialFunctions(self.W)

        return

    # Define problem form:
    def define_form(self, THETA, stationary):

        # Specify trial functions based on time discretization:
        self.v_tr, self.p_tr = (THETA==0.0)*self._v + (THETA!=0)*self.v, (THETA==0)*self._p + (THETA!=0)*self.p
        
        # Incompressibility
        self.F += self.b(self.p_, self.v_tr)

        # Add pressure term to the form:
        self.F += -self.b(self.p_tr, self.v_)

        # Add viscosity term to the form:
        self.F += self.a(self.v_tr, self.v_, self.var)

        # Add convective term to the form (fully non-linear):
        self.F += (1 - THETA)*self.c(self.v_tr, self.v_k, self.v_, self.var) \
                  + THETA*self.c(self.v_tr, self.v_tr, self.v_, self.var)

        # Add right-hand side:
        self.F += - self.d(self.v_, self.var)

        # Add natural boundary conditions:
        self.F += sum(self.bcs_natural)

        if not stationary:

            # Add discretized time derivative for non-stationary problems:
            self.F += self.rho(self.var)*dolfin.inner((self.v_tr - self.v_k)/self.dt, self.v_)*self.d_w*self.dx

        # Add stabilization (SUPG/grad-div, taken from [1]):
        if False:

            # Define weight of stabilization:
            h = dolfin.CellDiameter(self.V.mesh())
            delta_supg = 1/(1/float(self.dt) + 4*(prm.mu_l/prm.rho_l)/h**2)

            delta_supg = h**2#/(4*prm.mu_l/prm.rho_l)

            # Stabilize by bilinear part of the strong residuum:
            self.F += dolfin.inner(- self.mu(self.var)*dolfin.div(dolfin.grad(self.v_tr)) \
                                   + self.rho(self.var)*dolfin.grad(self.v_tr)*self.v_k \
                                   + dolfin.grad(self.p_tr),
                                   delta_supg*self.rho(self.var)*dolfin.grad(self.v_)*self.v_k)*self.d_w*self.dx

            # Stabilize by right hand side of the strong residuum:
            self.F += - self.d(delta_supg*self.rho(self.var)*dolfin.grad(self.v_)*self.v_k, self.var)

            # Add grad-div stabilization:
            # self.F += dolfin.inner(dolfin.div(self.v_tr), dolfin.div(self.v_))*self.d_w*self.dx

        return

    # Material parameters
    # -------------------

    # Density:
    def rho(self, *var):

        return prm.rho_l

    # Dynamic viscosity:
    def mu(self, *var):

        return prm.mu_l
    

    # Variational forms 
    # -----------------
    
    # Viscous stress tensor
    def S(self, u, *var):

        D = dolfin.sym(dolfin.grad(u))

        return 2*self.mu(*var)*D

    # Define discretized Cauchy stress tensor:
    def sigma(self, q, u, *var):

        # Identity tensor:
        I = dolfin.Identity(self.x.geometric_dimension())

        return -q*I + self.S(u, *var)

    # Divergence operator
    def b(self, q, u):

        return q*dolfin.div(u)*self.d_w*self.dx

    # Viscosity term:
    def a(self, u, v, *var):

        return dolfin.inner(self.S(u, *var), dolfin.grad(v))*self.d_w*self.dx

    # Convective term (Navier-Stokes form):
    def c(self, u, u_conv, v, *var):

        return self.rho(*var)*dolfin.inner(dolfin.grad(u)*u_conv,v)*self.d_w*self.dx

    # Right-hand side:
    def d(self, u, *var):

        # Define special density for buyoancy force:
        self.rho_buoy = self.rho

        # Gravitation body force:
        f = self.rho_buoy(*var)*prm.g*(-ez)

        return dolfin.inner(f,u)*self.d_w*self.dx

    # End of NavierStokesCartesian

# Formulate Navier-Stokes in h-axisymmetric geometry (with horizontal axis of symmetry):
class NavierStokesHAxisym(NavierStokesCartesian):

    def __init__(self, geometry):

        # Initialize superstructure:
        NavierStokesCartesian.__init__(self, geometry)

        # Set measure weight for h-axisymmetric geometry:
        self.d_w = self.x[1]

        return

    # Redefine forms
    # --------------

    # Divergence operator in horizontal-axisym cylindrical coordinates:
    def b(self, q, u):

        return q*(dolfin.div(u)*self.d_w + u[1])*self.dx

    # Viscosity term in horizontal-axisym cylindrical coordinates:
    def a(self, u, v, *var):

        return (dolfin.inner(NavierStokesCartesian.S(self, u, *var), dolfin.grad(v)) \
                + 2*self.mu(*var)*u[1]*v[1]/(self.x[1]*self.x[1]))*self.x[1]*self.dx

    # End of NavierStokesAxisym

# Formulate Navier-Stokes in v-axisymmetric geometry (with vertical axis of symmetry):
class NavierStokesVAxisym(NavierStokesCartesian):

    def __init__(self, geometry):

        # Initialize superstructure:
        NavierStokesCartesian.__init__(self, geometry)

        # Set measure weight for h-axisymmetric geometry:
        self.d_w = self.x[0]

        return

    # Redefine forms
    # --------------

    # Divergence operator in v-axisym cylindrical coordinates:
    def b(self, q, u):

        return q*(dolfin.div(u)*self.d_w + u[0])*self.dx

    # Viscosity term in v-axisym cylindrical coordinates:
    def a(self, u, v, *var):

        return (dolfin.inner(NavierStokesCartesian.S(self, u, *var), dolfin.grad(v)) \
                + 2*self.mu(*var)*u[0]*v[0]/(self.x[0]*self.x[0]))*self.x[0]*self.dx

    # End of NavierStokesAxisym

# Formulate rigid body motion in viscous fluid problem in body's reference frame:
class RigidBodyMotionVAxisymNoninertial(NavierStokesVAxisym):

    def __init__(self, geometry, **rb_properties):

        # Initialize Navier Stokes problem in v-axisym geometry
        NavierStokesVAxisym.__init__(self, geometry)

        # Define finite element for rigid body vertical velocity:
        R_ele = dolfin.FiniteElement("R", geometry.mesh.ufl_cell(), 0)

        # Add rigid body motion element:
        W_ele = dolfin.MixedElement(self.W.ufl_element().sub_elements() + [R_ele])

        # Define augmented FE space:
        self.W = dolfin.FunctionSpace(geometry.mesh, W_ele)

        # Redefine functions:
        self.w = dolfin.Function(self.W); self.v, self.p, self.vb = dolfin.split(self.w)
        self.w_k = dolfin.Function(self.W); self.v_k, self.p_k, self.vb_k = dolfin.split(self.w_k)

        # Redefine test and trial functions:
        self.v_, self.p_, self.vb_ = dolfin.TestFunctions(self.W)
        self._v, self._p, self._vb = dolfin.TrialFunctions(self.W)

        # Save rigid body properties:
        self.rb_mass = rb_properties["mass"]; self.rb_surface_id = rb_properties["srf_id"]

        return

    def define_form(self, THETA, stationary):

        self.vb_tr = (THETA==0.0)*self._vb + (THETA!=0)*self.vb

        NavierStokesVAxisym.define_form(self, THETA, stationary)

        # Add ODE for rigid body vertical velocity:
        for srf_id in self.rb_surface_id:
            
            self.F +=  1/dolfin.assemble(1.0*self.dx)\
                       *dolfin.inner(self.rb_mass*(-(self.vb_tr - self.vb_k)/self.dt + prm.g), self.vb_)*self.dx\
                       - (2*dolfin.pi)*dolfin.inner(dolfin.dot(self.sigma(self.p_tr, self.v_tr, self.var),-self.n)[1],
                                                    self.vb_)\
                                                    *self.d_w*self.ds(srf_id) # Check if 5 is body surface!

    # Add non-inertial correction to the right hand side:
    def d(self, u, *var):

        # Add non-inertial correction to the right hand side (check the sign):
        return NavierStokesVAxisym.d(self, u, *var) \
            + self.rho_buoy(*var)*dolfin.inner((self.vb - self.vb_k)/self.dt, u[1])*self.d_w*self.dx

    # this works ok:
    #return super().d(u) - self.rho(*var)*dolfin.inner((self.vb - self.vb_k)/self.dt, u[1])*self.d_w*self.dx

    # End of RigidBodyMotionaVAxisymNoninertial

# Formulate rigid body motion problem in Cartesian coordinates with ALE method:
class RigidBodyMotionALE(NavierStokesCartesian):

    pass

# END of Mechanical problems
# =======================

# (II) Thermal problems
# ---------------------

# Heat equation in Cartesian coordinates:
class FourierCartesian(Problem):

    def __init__(self, geometry):

        # Initialize general superstructure (form, boundary conditions, functions, geometry):
        Problem.__init__(self, geometry)

        # (I) Initialize FE spaces
        # ------------------------
        
        # Define function space:
        self.T = CGdeg(geometry.mesh)

        # Define temperature and enthalpy functions:
        self.theta = dolfin.Function(self.T); self.h = dolfin.Function(self.T)

        # Functions to store solutions from previous steps:
        self.theta_k = dolfin.Function(self.T); self.theta_kminus1 = dolfin.Function(self.T)

        # Test functions and trial functions:
        self.theta_ = dolfin.TestFunction(self.T); self._theta = dolfin.TrialFunction(self.T)

        # Bind with the superstructure:
        self.w = self.theta; self.w_k = self.theta_k; self.w_kminus1 = self.theta_kminus1; self.W = self.T

        # Initialize convective velocity:
        self.v_conv = dolfin.Constant(np.zeros(self.x.geometric_dimension()))

        # (II) Initialize problem
        # -----------------------

        # Initialize problem form:
        self.F = 0.0

        # Initialize essential and natural boundary conditions:
        self.bcs_essential = []; self.bcs_natural = []

        # Define volumetric and surface measure weight for projected problems:
        self.d_w = dolfin.Constant(1.0)
        # ========================

        return

    def define_form(self, THETA, stationary):

        theta = (THETA == 0.0)*self._theta + (THETA != 0)*self.theta

        # Add elliptic term (Crank-Nicholson scheme):
        self.F += THETA*self.a(self.theta, theta, self.theta_) \
                  + (1.0 - THETA)*self.a(self.theta_k, theta, self.theta_)

        # Add natural boundary condtions (sign: (+ heat flux) dot (outer normal)):
        self.F += sum(self.bcs_natural)

        # Add time discretization for non-stationary problems:
        if not stationary:

            self.F += (THETA*self.rhoc(self.theta) + (1.0 - THETA)*self.rhoc(self.theta_k)) \
                      *(dolfin.inner(theta - self.theta_k, self.theta_)/self.dt \
                         + dolfin.inner(self.v_conv, dolfin.grad(theta))*self.theta_)*self.d_w*self.dx

        return

    # Material parameters
    # -------------------

    # Define material parameters (volumetric heat capacity):
    def rhoc(self, theta):

        return prm.rho_s*prm.c_s
    
    # Define material parameters (heat conductivity):
    def k(self, theta):

        return prm.k_s

    # Variational forms
    # -----------------

    # Elliptic variational term:
    def a(self, theta1, theta2, vartheta):

        return self.k(theta1)*dolfin.inner(dolfin.grad(theta2), dolfin.grad(vartheta))*self.d_w*self.dx

    # END of FourierCartesian

class FourierVAxisym(FourierCartesian):

    def __init__(self, geometry):

        FourierCartesian.__init__(self, geometry)

        self.d_w = self.x[0]

        return

    # END of FourierVAxisym

# (END) Thermal problems
# ======================

# General Data structure
# ----------------------

class Data:

    # pri inicializaci chci loggovat, vytvorit soubory xdmf a npy
    def __init__(self, geometry, path = "./out/data/"):

        # Save path to data folder:
        self.path = path

        # Save communicator rank:
        self.rank = geometry.rank

        # Initialize XDMF data files:
        self.data_xdmf = dolfin.XDMFFile(geometry.comm,
                                         path + "data_FEM.xdmf")

        # Set xdmf file parameters
        
        # This enables one to check files while computation is still running:
        self.data_xdmf.parameters["flush_output"] = True

        # This optimizes saving and save space (single mesh):
        self.data_xdmf.parameters["functions_share_mesh"] = True

        # Initialize py data file and its basic structure:
        self.data_py = {}; self.data_py["time_series"] = {"t": np.asarray([])}; self.data_py["parameters"] = {}
              
        return

    def save_xdmf(self, t, *data):

        # Save material params
        for file_ in data:

            # Save function to xdmf file (t=t as a third argument is necessary due to pybind11 bug):
            self.data_xdmf.write(file_, t = t)

        return

    def save_time_series(self, t, **data):

        # Write time point:
        self.data.data_py["time_series"]["t"] = np.append(self.data.data_py["time_series"]["t"], t)

        # Write python series data:
        for val in data:

            self.data.data_py["time_series"][val] = np.append(self.data.data_py["time_series"][val], data[val])

    def save_py(self):

        if self.rank == 0:

            # Save python file only on single rank:
            np.save(self.path + 'data_py.npy', self.data_py)

        return

    def close(self):

        # Close FEM data files:
        self.data_xdmf.close()

        # Save and close py data files:
        self.save_py()

        return

# (END) General Data structure
# ===========================


