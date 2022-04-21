# Sem bych chtel nastrkat veskery kod, ktery se tyka formulace problemu ve frniuvsd

import dolfin

import em.enthalpy_method as em
import sim.params as prm

# THETA scheme time discretization parameter (0.0 for explicit):
THETA = 0.0

# FE degree (for mechanical Taylor-Hood, for temperature +1):
DEGREE = 1

# Default time step size:
DT = dolfin.Constant(1e-3)

# Define unit vector pointing upwards:
ez = dolfin.Constant((0.0, 1.0))

# -------------------
# Material parameters
# -------------------

# Density:
def rho(theta, eps_minus = em.EPS, eps_plus = em.EPS, deg = em.DEG):

    return em.mollify(prm.rho_l,
                      prm.rho_s,
                      theta,
                      x0 = prm.theta_m,
                      eps_minus = eps_minus,
                      eps_plus = eps_plus,
                      deg = deg)

# Dynamic viscosity:
def mu(theta, eps_minus = em.EPS, eps_plus = em.EPS, deg = em.DEG):

    return em.mollify(prm.mu_l,
                      prm.mu_s,
                      theta,
                      x0 = prm.theta_m,
                      eps_minus = eps_minus,
                      eps_plus = eps_plus,
                      deg = deg)

# Volumetric heat capacity
def rhoc(theta, eps_minus = em.EPS, eps_plus = em.EPS, deg = em.DEG):

    # Return total effective rho*c: 
    return em.mollify(prm.rho*prm.c_s,
                      prm.rho*prm.c_l,
                      theta,
                      x0 = prm.theta_m,
                      eps_minus = eps_minus,
                      eps_plus = eps_plus,
                      deg = deg) \
                      + em.impulse(prm.rho*prm.L_m,
                                   theta,
                                   x0 = prm.theta_m,
                                   eps_minus = eps_minus,
                                   eps_plus = eps_plus,
                                   deg = deg)

# Heat conductivity:
def k(theta, eps_minus = em.EPS, eps_plus = em.EPS, deg = em.DEG):

    return em.mollify(prm.k_s,
                      prm.k_l,
                      theta,
                      x0 = prm.theta_m,
                      eps_minus = eps_minus,
                      eps_plus = eps_plus,
                      deg = deg)

# ===================

# General variational problem:
class Problem:

    def __init__(self):

        # Define default time step size:
        self.dt = DT

        # Initialize unknown functions:
        self.w = None; self.w_k = None; self.w_k_minus1 = None

        # Initialize form:
        self.F = 0

        # Initialize boundary conditions:
        self.bcs_essential = []; self.bcs_natural = []

    def define_solver(self, THETA = THETA):

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

    def assign_next_step(self):

        self.w_k_minus1.assign(self.w_k)

        self.w_k.assign(self.w)

        return

# (I) Mechanical problems
# -----------------------

# Build Navier-Stokes problem formulation in cartesian geometry: 
class NavierStokesCartesian(Problem):

    def __init__(self, geometry):

        # Initialize superstructure (form, boundary conditions, functions):
        Problem.__init__(self)

        # Define measures, measure weight, and normal:
        self.dx = geometry.dx; self.ds = geometry.ds; self.d_w = 1.0; self.n = geometry.n

        # Define spatial coordinate:
        self.x = dolfin.SpatialCoordinate(geometry.mesh)

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
        self.w_k_minus1 = dolfin.Function(self.W)

        # Define test and trial functions:
        self.v_, self.p_ = dolfin.TestFunctions(self.W); self._v, self._p = dolfin.TrialFunctions(self.W)

        return

    # Define problem form:
    def define_form(self, THETA, stationary = False):

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

    def define_solver(self, THETA = THETA):

        # Update form:
        self.define_form(THETA)

        # Define solver using superstructure method:
        Problem.define_solver(self, THETA)

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

    def define_form(self, THETA):

        NavierStokesVAxisym.define_form(self, THETA)

        # Add ODE for rigid body vertical velocity:
        self.F +=  1/dolfin.assemble(1.0*self.dx)\
                   *dolfin.inner(prm.sphere_mass*(-(self.vb - self.vb_k)/self.dt + prm.g), self.vb_)*self.dx\
                   - (2*dolfin.pi)*dolfin.inner(dolfin.dot(self.sigma(self.p,self.v),-self.n)[1],self.vb_)\
                   *self.d_w*self.ds(5) # Check if 5 is body surface!

    # Add non-inertial correction to the right hand side:
    def d(self, u):

        return super().d(u) - dolfin.inner((self.vb - self.vb_k)/self.dt, u[1])*self.d_w*self.dx

    # End of RigidBodyMotionaVAxisymNoninertial

# Formulate rigid body motion problem in Cartesian coordinates with ALE method:
class RigidBodyMotionALE(NavierStokesCartesian):

    pass

# END of Mechanical problems
# =======================

# (II) Thermal problems
# ---------------------

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
        for field in data:

            self.data_xdmf[field].write(data[field], t)

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


