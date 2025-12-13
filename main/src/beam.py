import numpy as np
import scipy as sp
from sympy import *
from matplotlib import pyplot as plt
from matplotlib import animation

class Beam:
    """
    IBVP setup for the damped Euler–Bernoulli beam:
        rhoA * w_tt + cd * w_t + EI * w_xxxx = q(x,t)
    with clamped-free BCs:
        w(0,t)=0, w_x(0,t)=0, w_xx(L,t)=0, w_xxx(L,t)=0
    and ICs:
        w(x,0)=alpha*phi1(x),  w_t(x,0)=0.

    Spatial discretization: uniform grid + 5-pt stencil for w_xxxx on interior nodes.
    BC handling: "BC row enforcement" implemented via elimination (reconstruction map T).
    Time stepping: trapezoid on first-order system y' = A y + b(t).
    """

    # -------------------------
    # Construction / parameters
    # -------------------------
    def __init__(
        self,
        L=1.0,
        m=101,
        rhoA=1.0,
        EI=1.0,
        cd=0.0,
        alpha=1.0,
        forcing_type="step",      # "step" or "tip"
        forcing_params=None,      # dict of parameters for the load
        phi1_type="cantilever",   # currently only "cantilever"
    ):
        self.L = float(L)
        self.m = int(m)
        self.rhoA = float(rhoA)
        self.EI = float(EI)
        self.cd = float(cd)
        self.alpha = float(alpha)

        self.forcing_type = forcing_type
        self.forcing_params = forcing_params or {}
        self.phi1_type = phi1_type

        # build grids
        self.x_full = np.linspace(0.0, self.L, self.m)
        self.dx = self.x_full[1] - self.x_full[0]

        # reduced grid (unknowns evolved at j=2..m-3)
        self.x_red = self.x_full[2:-2]
        self.n = self.m - 4  # reduced DOF count

        if self.m < 8:
            raise ValueError("Need at least m>=8 grid points for chosen one-sided stencils.")

        # build BC elimination map and reduced operators/system
        self.T = self._clamped_free_elimination_matrix(self.m)
        self.D4_red = self._build_D4_red(self.m, self.dx, self.T)
        self.Acal = self._assemble_Acal(self.D4_red)

        # set forcing + IC shape
        self.q_func = self._make_q_func(self.forcing_type, self.forcing_params)
        self.bcal = self._make_bcal(self.q_func)
        self.phi1 = self._make_phi1(self.phi1_type)

    # -------------------------
    # Mode shape (IC)
    # -------------------------
    @staticmethod
    def phi1_cantilever(x, L):
        """
        First clamped-free bending mode (up to scaling):
            phi = cosh(beta x) - cos(beta x) - sigma (sinh(beta x) - sin(beta x))
        with beta1 ≈ 1.8751, beta = beta1/L
        Normalized so phi(L)=1.
        """
        beta1 = 1.875104068711961
        beta = beta1 / L
        sigma = (np.cosh(beta1) + np.cos(beta1)) / (np.sinh(beta1) + np.sin(beta1))

        phi = (
            np.cosh(beta * x)
            - np.cos(beta * x)
            - sigma * (np.sinh(beta * x) - np.sin(beta * x))
        )

        phi_L = (
            np.cosh(beta * L)
            - np.cos(beta * L)
            - sigma * (np.sinh(beta * L) - np.sin(beta * L))
        )

        return phi / phi_L

    def _make_phi1(self, phi1_type):
        if phi1_type.lower() == "cantilever":
            return lambda x: self.phi1_cantilever(x, self.L)
        raise ValueError(f"Unknown phi1_type='{phi1_type}'. Currently supports: 'cantilever'.")

    # -------------------------
    # Forcing q(x,t)
    # -------------------------
    @staticmethod
    def q_step_factory(q0, t0=0.0):
        """
        Uniform step:
            q(x,t) = q0 * H(t - t0)
        """
        def q_func(x, t):
            return q0 * (t >= t0) * np.ones_like(x)
        return q_func

    @staticmethod
    def indicator_tip_region(x, L, dx_tip):
        """
        Indicator for [L-dx_tip, L]
        """
        return ((x >= (L - dx_tip)) & (x <= L)).astype(float)

    @staticmethod
    def qimp_square(q_amp, t_start, t_width):
        """
        Square pulse in time.
        """
        def qimp(t):
            return q_amp * ((t >= t_start) & (t <= t_start + t_width))
        return qimp

    @staticmethod
    def qimp_gaussian(q_amp, t_center, sigma):
        """
        Smooth Gaussian pulse in time.
        """
        def qimp(t):
            return q_amp * np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
        return qimp

    def q_tip_factory(self, dx_tip, qimp):
        """
        Tip-localized load:
            q(x,t) = qimp(t) * 1_{[L-dx_tip, L]}(x)
        """
        def q_func(x, t):
            return qimp(t) * self.indicator_tip_region(x, self.L, dx_tip)
        return q_func

    def _make_q_func(self, forcing_type, params):
        ft = forcing_type.lower()

        if ft == "step":
            # expected: q0, optional: t0
            q0 = float(params.get("q0", 1.0))
            t0 = float(params.get("t0", 0.0))
            return self.q_step_factory(q0=q0, t0=t0)

        if ft == "tip":
            # expected: dx_tip plus pulse parameters
            dx_tip = float(params.get("dx_tip", 0.05 * self.L))

            pulse = params.get("pulse", "gaussian").lower()
            if pulse == "square":
                q_amp = float(params.get("q_amp", 1.0))
                t_start = float(params.get("t_start", 0.1))
                t_width = float(params.get("t_width", 0.01))
                qimp = self.qimp_square(q_amp=q_amp, t_start=t_start, t_width=t_width)
            elif pulse == "gaussian":
                q_amp = float(params.get("q_amp", 1.0))
                t_center = float(params.get("t_center", 0.1))
                sigma = float(params.get("sigma", 0.01))
                qimp = self.qimp_gaussian(q_amp=q_amp, t_center=t_center, sigma=sigma)
            else:
                raise ValueError("For tip forcing, params['pulse'] must be 'square' or 'gaussian'.")

            return self.q_tip_factory(dx_tip=dx_tip, qimp=qimp)

        raise ValueError("forcing_type must be 'step' or 'tip'.")

    # -------------------------
    # BC elimination: clamped-free
    # -------------------------
    @staticmethod
    def _clamped_free_elimination_matrix(m: int) -> np.ndarray:
        """
        Build T such that W_full = T @ W_red enforces:
            w0 = 0
            w_x(0)=0 using (-3 w0 + 4 w1 - w2)/(2 dx)=0  => w1 = w2/4 (since w0=0)
            w_xx(L)=0 using (2 w_{m-1} - 5 w_{m-2} + 4 w_{m-3} - w_{m-4})/dx^2=0
            w_xxx(L)=0 using (5 w_{m-1} - 18 w_{m-2} + 24 w_{m-3} - 14 w_{m-4} + 3 w_{m-5})/(2 dx^3)=0

        Reduced unknowns:
            W_red = [w2, w3, ..., w_{m-3}]  length n=m-4
        """
        if m < 8:
            raise ValueError("Need at least m>=8 grid points for chosen one-sided stencils.")

        n = m - 4
        T = np.zeros((m, n))

        # Map reduced entries directly: for j=2..m-3, wj = W_red[j-2]
        for j in range(2, m-2):  # j=2..m-3
            T[j, j-2] = 1.0

        # w0 = 0 already handled (row 0 all zeros)

        # w_x(0)=0 -> w1 = w2/4, where w2 is reduced index 0
        T[1, 0] = 0.25

        # Tip BCs: solve for w_{m-2}, w_{m-1} in terms of w_{m-3}, w_{m-4}, w_{m-5}
        # Let a=w_{m-1}, b=w_{m-2}, c=w_{m-3}, d=w_{m-4}, e=w_{m-5}.
        # b = (28c - 23d + 6e)/11
        # a = (5b - 4c + d)/2
        ic = n - 1  # c index in reduced
        id_ = n - 2 # d
        ie = n - 3  # e

        # b row (m-2)
        T[m-2, ic] = 28.0 / 11.0
        T[m-2, id_] = -23.0 / 11.0
        T[m-2, ie] = 6.0 / 11.0

        # a row (m-1) = (5/2)*b + (-4c + d)/2
        T[m-1, :] += (5.0 / 2.0) * T[m-2, :]
        T[m-1, ic] += -4.0 / 2.0
        T[m-1, id_] += 1.0 / 2.0

        return T

    @staticmethod
    def _build_D4_red(m: int, dx: float, T: np.ndarray) -> np.ndarray:
        """
        Build reduced D4 operator using interior 5-pt stencil rows j=2..m-3,
        then substitute W_full = T W_red:
            D4_red = D4_full[rows,:] @ T
        """
        D4_full = np.zeros((m, m))
        for j in range(2, m-2):  # interior rows
            D4_full[j, j-2] = 1.0
            D4_full[j, j-1] = -4.0
            D4_full[j, j]   = 6.0
            D4_full[j, j+1] = -4.0
            D4_full[j, j+2] = 1.0
        D4_full /= dx**4

        rows = np.arange(2, m-2)  # j=2..m-3, length n=m-4
        D4_red = D4_full[rows, :] @ T
        return D4_red

    # -------------------------
    # Assemble method-of-lines system
    # -------------------------
    def _assemble_Acal(self, D4_red: np.ndarray) -> np.ndarray:
        """
        First-order system y' = Acal y + b(t), where y=[W_red; V_red].
        """
        n = self.n
        K = (self.EI / self.rhoA) * D4_red
        gamma = self.cd / self.rhoA

        Z = np.zeros((n, n))
        I = np.eye(n)

        Acal = np.block([
            [Z,  I],
            [-K, -gamma * I]
        ])
        return Acal

    def _make_bcal(self, q_func):
        """
        b(t) for y' = A y + b(t):
            W' = V
            V' = (1/rhoA)(Q(t) - cd V - EI D4 W)  -> forcing part is Q(t)/rhoA
        Here Q is evaluated on reduced nodes x_red.
        """
        def bcal(t):
            Q = q_func(self.x_red, t)
            return np.concatenate([np.zeros(self.n), Q / self.rhoA])
        return bcal

    # -------------------------
    # Initial condition vector
    # -------------------------
    def initial_state(self):
        """
        Return y0 = [W0_red; V0_red] for:
            W0_red = alpha * phi1(x_red)
            V0_red = 0
        """
        W0_red = self.alpha * self.phi1(self.x_red)
        V0_red = np.zeros_like(W0_red)
        return np.concatenate([W0_red, V0_red])

    # -------------------------
    # Reconstruction (full field from reduced)
    # -------------------------
    def reconstruct_full(self, W_red):
        """
        Reconstruct full nodal W_full on x_full from reduced W_red using W_full = T W_red.
        """
        return self.T @ W_red

    # -------------------------
    # Trapezoid time marching
    # -------------------------
    def solve_trapezoid(self, dt, Tfinal, store_every=1):
        """
        March y' = A y + b(t) with trapezoid:
            (I - dt/2 A) y_{k+1} = (I + dt/2 A) y_k + dt/2 (b_k + b_{k+1})

        Returns:
            t_hist (Nsamples,)
            W_full_hist (Nsamples, m)  full deflection snapshots
            V_red_hist (Nsamples, n)   reduced velocity snapshots (optional usefulness)
        """
        dt = float(dt)
        Tfinal = float(Tfinal)

        y = self.initial_state()
        N = y.size

        I = np.eye(N)
        LHS = I - 0.5 * dt * self.Acal
        RHS_mat = I + 0.5 * dt * self.Acal

        Nt = int(np.round(Tfinal / dt))
        if Nt < 1:
            raise ValueError("Tfinal must be >= dt.")

        t_hist = []
        W_full_hist = []
        V_red_hist = []

        def record(t, yvec):
            W_red = yvec[:self.n]
            V_red = yvec[self.n:]
            W_full = self.reconstruct_full(W_red)
            t_hist.append(t)
            W_full_hist.append(W_full)
            V_red_hist.append(V_red)

        record(0.0, y)

        for k in range(Nt):
            tk = k * dt
            tk1 = (k + 1) * dt

            rhs = RHS_mat @ y + 0.5 * dt * (self.bcal(tk) + self.bcal(tk1))
            y = np.linalg.solve(LHS, rhs)

            if ((k + 1) % store_every) == 0:
                record(tk1, y)

        return np.array(t_hist), np.array(W_full_hist), np.array(V_red_hist)
    
    # Add this method inside your Beam class (same indentation level as solve_trapezoid)

    def solve_backward_euler(self, dt, Tfinal, store_every=1):
        """
        March y' = A y + b(t) with Backward Euler:
            y_{k+1} = y_k + dt * (A y_{k+1} + b(t_{k+1}))
        =>
            (I - dt A) y_{k+1} = y_k + dt * b(t_{k+1})

        Returns:
            t_hist (Nsamples,)
            W_full_hist (Nsamples, m)  full deflection snapshots
            V_red_hist (Nsamples, n)   reduced velocity snapshots
        """
        dt = float(dt)
        Tfinal = float(Tfinal)

        y = self.initial_state()
        N = y.size

        I = np.eye(N)
        LHS = I - dt * self.Acal

        Nt = int(np.round(Tfinal / dt))
        if Nt < 1:
            raise ValueError("Tfinal must be >= dt.")

        t_hist = []
        W_full_hist = []
        V_red_hist = []

        def record(t, yvec):
            W_red = yvec[:self.n]
            V_red = yvec[self.n:]
            W_full = self.reconstruct_full(W_red)
            t_hist.append(t)
            W_full_hist.append(W_full)
            V_red_hist.append(V_red)

        record(0.0, y)

        for k in range(Nt):
            tk1 = (k + 1) * dt
            rhs = y + dt * self.bcal(tk1)
            y = np.linalg.solve(LHS, rhs)

            if ((k + 1) % store_every) == 0:
                record(tk1, y)

        return np.array(t_hist), np.array(W_full_hist), np.array(V_red_hist)


    # -------------------------
    # Quick plotting utility
    # -------------------------
    def plot_snapshot(self, W_full, title="Beam deflection snapshot"):
        plt.figure()
        plt.plot(self.x_full, W_full)
        plt.xlabel("x")
        plt.ylabel("w(x,t)")
        plt.title(title)
        plt.grid(True)
        plt.show()

    def animate_deflection(
        self,
        t_hist,
        W_full_hist,
        interval=30,
        ylim=None,
        file_name=None
    ):
        """
        Animate transient beam deflection w(x,t).

        Parameters
        ----------
        t_hist : (Nt,) array
            Time history.
        W_full_hist : (Nt, m) array
            Full deflection snapshots.
        interval : int
            Delay between frames in ms.
        ylim : tuple or None
            y-axis limits (ymin, ymax). If None, auto-scaled.
        file_name : str or None
            If provided, saves animation (e.g. 'beam.gif').
        """

        fig, ax = plt.subplots()
        line, = ax.plot(self.x_full, W_full_hist[0], lw=2)

        ax.set_xlabel("x")
        ax.set_ylabel("w(x,t)")
        ax.set_title("Beam deflection")

        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            wmax = np.max(np.abs(W_full_hist))
            ax.set_ylim(-1.2*wmax, 1.2*wmax)

        ax.grid(True)

        def update(frame):
            line.set_ydata(W_full_hist[frame])
            ax.set_title(f"Beam deflection, t = {t_hist[frame]:.3f} s")
            return (line,)

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(t_hist),
            interval=interval,
            blit=True
        )
        if file_name is None:
            file_name = "untitled_beam_animation.gif"
            
        save_path = "../data/" + file_name
        anim.save(save_path, writer="pillow", fps=1000//interval)

        plt.show()
        print("Saved animation to:", save_path)
        return anim
    
    def plot_wingtip_amplitude(
        self,
        t_hist,
        W_full_hist,
        title="Wingtip deflection vs time",
    ):
        """
        Plot wingtip deflection w(L,t) over time.

        Parameters
        ----------
        t_hist : (Nt,) array
            Time history.
        W_full_hist : (Nt, m) array
            Full deflection snapshots.
        title : str
            Plot title.
        """
        # Wingtip is the last spatial node
        w_tip = W_full_hist[:, -1]

        plt.figure()
        plt.plot(t_hist, w_tip, lw=2)
        plt.xlabel("Time t [s]")
        plt.ylabel("Wingtip deflection w(L,t) [m]")
        plt.title(title)
        plt.grid(True)
        plt.show()
