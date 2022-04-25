# Sem bych chtel nastrkat veskery kod, ktery se tyka formulace problemu ve frniuvsd

import dolfin

import sim.params as prm

# THETA scheme time discretization parameter (0.0 for explicit):
THETA = 0.0

# FE degree (for mechanical Taylor-Hood, for temperature +1):
DEGREE = 1

# Default time step size:
DT = dolfin.Constant(1e-3)

# Define unit vector pointing upwards:
ez = dolfin.Constant((0.0, 1.0))

def CGdeg(mesh, deg = DEGREE):

    return dolfin.FunctionSpace(mesh, dolfin.FiniteElement("CG", mesh.ufl_cell(), deg))

def TaylorHood(mesh, deg = DEGREE):

    return

def bubble(mesh, deg = DEGREE):

    return

# General variational problem:
class Problem:

    def __init__(self, geometry):

        # Define default time step size:
        self.dt = DT

        # Define measures, measure weight, and normal:
        self.dx = geometry.dx; self.ds = geometry.ds; self.d_w = 1.0; self.n = geometry.n

        # Define spatial coordinate:
        self.x = dolfin.SpatialCoordinate(geometry.mesh)

        # Initialize FE space and functions:
        self.W = None; self.w = None; self.w_k = None; self.w_kminus1 = None

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

    # Set initial condition (from expression):
    def set_initial(self, w_initial):

        # Use theta_initial expression to interpolate the function:
        self.w_k.assign(dolfin.interpolate(w_initial, self.W))

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
        V_ele = dolfin.VectorElement("CG", geometry.mesh.ufl_cell(), DEGREE+1)
        P_ele = dolfin.FiniteElement("CG", geometry.mesh.ufl_cell(), DEGREE)

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
        v, p = (THETA == 0.0)*self._v + (THETA != 0)*self.v, (THETA == 0)*self._p + (THETA!=0)*self.p

        # Incompressibility
        self.F += self.b(self.p_, v)

        # Add pressure term to the form:
        self.F += -self.b(p, self.v_)

        # Add viscosity term to the form:
        self.F += self.a(v, self.v_)

        # Add convective term to the form (fully non-linear):
        self.F += (1 - THETA)*self.c(v, self.v_k, self.v_) + THETA*self.c(v, v, self.v_)

        # Add right-hand side:
        self.F += - self.d(self.v_)

        # Add natural boundary conditions:
        self.F += sum(self.bcs_natural)

        if not stationary:

            # Add discretized time derivative for non-stationary problems:
            self.F += self.rho()*dolfin.inner((v - self.v_k)/self.dt, self.v_)*self.d_w*self.dx

        return

    # ------------------------------------
    # Problem specific material parameters
    # ------------------------------------

    # Density:
    def rho(self, theta = None):

        return prm.rho_l

    # Dynamic viscosity:
    def mu(self, theta = None):

        return prm.mu_l
    
    # ----------------------
    # Problem specific forms
    # ----------------------
    
    # Viscous stress tensor
    def S(self, u):

        D = dolfin.sym(dolfin.grad(u))

        return 2*self.mu()*D

    # Define discretized Cauchy stress tensor:
    def sigma(self, q, u):

        # Identity tensor:
        I = dolfin.Identity(self.x.geometric_dimension())

        return -q*I + self.S(u)

    # Divergence operator
    def b(self, q, u):

        return q*dolfin.div(u)*self.d_w*self.dx

    # Viscosity term:
    def a(self, u, v):

        return dolfin.inner(self.S(u), dolfin.grad(v))*self.d_w*self.dx

    # Convective term (Navier-Stokes form):
    def c(self, u, u_k, v):

        return self.rho()*dolfin.inner(dolfin.grad(u)*u_k,v)*self.d_w*self.dx

    # Right-hand side:
    def d(self, u):

        # Gravitation body force:
        f = self.rho()*prm.g*(-ez)

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
    def a(self, u, v):

        return (dolfin.inner(NavierStokesCartesian.S(self, u), dolfin.grad(v)) \
                + 2*self.mu()*u[1]*v[1]/(self.x[1]*self.x[1]))*self.x[1]*self.dx

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
    def a(self, u, v):

        return (dolfin.inner(NavierStokesCartesian.S(self, u), dolfin.grad(v)) \
                + 2*self.mu()*u[0]*v[0]/(self.x[0]*self.x[0]))*self.x[0]*self.dx

    # End of NavierStokesAxisym

# Formulate rigid body motion problem in body's reference frame:
class RigidBodyMotionVAxisymNoninertial(NavierStokesVAxisym):

    def __init__(self, geometry):

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
        self.v_, self.p_, self.vb_ = dolfin.TestFunctions(self.W); _v, _p, _vb = dolfin.TrialFunctions(self.W)

        return

    def define_form(self, THETA, stationary):

        NavierStokesVAxisym.define_form(self, THETA, stationary)

        # Add ODE for rigid body vertical velocity:
        self.F +=  1/dolfin.assemble(1.0*self.dx)\
                   *dolfin.inner(prm.sphere_mass*(-(self.vb - self.vb_k)/self.dt + prm.g), self.vb_)*self.dx\
                   - (2*dolfin.pi)*dolfin.inner(dolfin.dot(self.sigma(self.p,self.v),-self.n)[1],self.vb_)\
                   *self.d_w*self.ds(5) # Check if 5 is body surface!

    # Add non-inertial correction to the right hand side:
    def d(self, u):

        # Add non-inertial correction to the right hand side (check the sign):
        return NavierStokesVAxisym.d(self, u) + dolfin.inner((self.vb - self.vb_k)/self.dt, u[1])*self.d_w*self.dx

    # this works ok:
    #return super().d(u) - dolfin.inner((self.vb - self.vb_k)/self.dt, u[1])*self.d_w*self.dx

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

    # Define material parameters (volumetric heat capacity):
    def rhoc(self, theta):

        return prm.rho_l*prm.c_l
    
    # Define material parameters (heat conductivity):
    def k(self, theta):

        return prm.k_l

    # Elliptic variational term:
    def a(self, theta1, theta2, vartheta):

        return self.k(theta1)*dolfin.inner(dolfin.grad(theta2), dolfin.grad(vartheta))*self.d_w*self.dx

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
                      *dolfin.inner(theta - self.theta_k, self.theta_)/self.dt*self.d_w*self.dx

        return

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
    def __init__(self, path, geometry):

        # Save path to data files:
        self.path = path

        # Initialize XDMF data files:
        self.data_xdmf = {}

        # Initialize py data file:
        self.data_py = {}
              
        return

    def save_xdmf(self, t, **data):

        # Save material params
        for var in data:

            self.data_xdmf[var].write(data[var], t)

        return

    def save_py(self):
        
        np.save(self.path + 'data_py.npy', self.data_py)

        return

    def close(self):

        # Close FEM data files:
        for xdmf in self.data_xdmf:
            
            self.xdmf.close()

        # Save and close py data files:
        self.save_py()

        return

# (END) General Data structure
# ============================


