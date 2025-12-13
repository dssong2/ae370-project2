import numpy as np
import scipy as sp
from sympy import *
from matplotlib import pyplot as plt

from .beam import Beam


class NumericalErrorAnalysis:
    """
    Convergence studies for the Beam solver.

    Temporal convergence:
      - Fix spatial grid (m large enough), vary dt.
      - Compare final-time solution to a reference computed with dt_ref << min(dt_list).
      - Expected: Backward Euler ~ O(dt^1), Trapezoid ~ O(dt^2)

    Spatial convergence (optional but recommended):
      - Fix dt very small, vary m.
      - Compare final-time solution to a fine-grid reference, interpolated onto coarse x.
      - Expected: ~O(dx^2) for the 5-pt 4th-derivative stencil used.
    """

    def __init__(self, beam: Beam):
        self.beam = beam

    # -------------------------
    # Utilities
    # -------------------------
    def clone_beam(self, **overrides) -> Beam:
        """Create a Beam with the same parameters as self.beam, with optional overrides."""
        b = self.beam
        params = dict(
            L=b.L,
            m=b.m,
            rhoA=b.rhoA,
            EI=b.EI,
            cd=b.cd,
            alpha=b.alpha,
            forcing_type=b.forcing_type,
            forcing_params=dict(b.forcing_params) if b.forcing_params is not None else None,
            phi1_type=b.phi1_type,
        )
        params.update(overrides)
        return Beam(**params)

    @staticmethod
    def rel_L2_discrete(x, u, v, eps=1e-14) -> float:
        """
        Relative discrete L2 norm:
            ||u - v||_2 / ||v||_2
        approximated via sqrt(dx * sum(.) ).
        """
        dx = x[1] - x[0]
        diff = u - v
        num = np.sqrt(dx * np.sum(diff**2))
        den = np.sqrt(dx * np.sum(v**2)) + eps
        return num / den

    def solve_final_W(self, method: str, dt: float, Tfinal: float, m_override=None):
        """
        Run Beam solver and return final full deflection W_full(Tfinal) and x_full.
        """
        if m_override is not None:
            beam = self.clone_beam(m=int(m_override))
        else:
            beam = self.beam

        # store_every very large -> only save initial + final (lightweight)
        Nt = int(np.round(Tfinal / dt))
        store_every = Nt  # record only at final step

        if method.lower() in ["trap", "trapezoid"]:
            t_hist, W_hist, _ = beam.solve_trapezoid(dt=dt, Tfinal=Tfinal, store_every=store_every)
        elif method.lower() in ["be", "backward_euler", "backward-euler", "backward"]:
            t_hist, W_hist, _ = beam.solve_backward_euler(dt=dt, Tfinal=Tfinal, store_every=store_every)
        else:
            raise ValueError("method must be 'trapezoid' or 'backward_euler'.")

        # last stored snapshot is the final time
        return beam.x_full, W_hist[-1], beam

    @staticmethod
    def estimate_order(x_vals, err_vals):
        """
        Estimate observed convergence order between consecutive points on a log-log curve:
            p_i = log(err_i/err_{i-1}) / log(h_i/h_{i-1})
        Returns p array with length len(err)-1.
        """
        x = np.array(x_vals, dtype=float)
        e = np.array(err_vals, dtype=float)
        # sort by x increasing
        idx = np.argsort(x)
        x = x[idx]
        e = e[idx]
        p = np.log(e[1:] / e[:-1]) / np.log(x[1:] / x[:-1])
        return x, e, p

    # -------------------------
    # Temporal convergence studies
    # -------------------------
    def temporal_convergence(
        self,
        method: str,
        dt_list,
        Tfinal: float,
        dt_ref_factor: int = 16,
        expected_order: int | None = None,
        make_plot: bool = True,
        title: str | None = None,
    ):
        """
        Temporal convergence test:
          - Fix m (current beam.m), compute reference at dt_ref = min(dt_list)/dt_ref_factor.
          - Compute relative L2 error at Tfinal against reference.
          - Plot loglog(error vs dt) + O(dt^p) guide.

        Returns dict with arrays and observed orders.
        """
        dt_list = np.array(dt_list, dtype=float)
        dt_min = float(np.min(dt_list))
        dt_ref = dt_min / float(dt_ref_factor)

        # Reference
        x_ref, W_ref, _ = self.solve_final_W(method=method, dt=dt_ref, Tfinal=Tfinal)

        errs = []
        for dt in dt_list:
            x, W, _ = self.solve_final_W(method=method, dt=float(dt), Tfinal=Tfinal)

            # grids should match since m is fixed
            if x.shape != x_ref.shape or np.max(np.abs(x - x_ref)) > 1e-12:
                raise RuntimeError("Temporal test expects same spatial grid (fixed m).")

            errs.append(self.rel_L2_discrete(x, W, W_ref))

        # estimate observed order
        x_sorted, e_sorted, p_obs = self.estimate_order(dt_list, errs)

        # pick expected order if not supplied
        if expected_order is None:
            if method.lower() in ["trap", "trapezoid"]:
                expected_order = 2
            else:
                expected_order = 1

        out = dict(
            method=method,
            dt_ref=dt_ref,
            dt=np.array(x_sorted),
            err=np.array(e_sorted),
            p_obs=np.array(p_obs),
            expected_order=expected_order
        )

        if make_plot:
            plt.figure()
            plt.loglog(out["dt"], out["err"], "o-", label="Relative L2 error at T")

            # reference slope line anchored to the largest dt point
            ref = out["err"][-1] * (out["dt"] / out["dt"][-1]) ** expected_order
            plt.loglog(out["dt"], ref, "--", label=f"O(Δt^{expected_order}) reference")

            plt.xlabel("Δt")
            plt.ylabel("‖e‖ (relative discrete L2)")
            if title is None:
                plt.title(f"Temporal convergence: {method} (m={self.beam.m}, T={Tfinal})")
            else:
                plt.title(title)
            plt.grid(True, which="both")
            plt.legend()
            plt.show()

            # print observed orders
            print(f"\nTemporal convergence ({method}):")
            print(f"  dt_ref = {dt_ref:.3e}")
            for i in range(len(out["p_obs"])):
                print(f"  p between dt={out['dt'][i]:.3e} and {out['dt'][i+1]:.3e} : {out['p_obs'][i]:.3f}")

        return out

    def run_temporal_convergence_suite(
        self,
        Tfinal: float,
        dt_list=None,
        dt_ref_factor: int = 16
    ):
        """
        Convenience wrapper: run both methods with the same dt_list.
        """
        if dt_list is None:
            # A reasonable default range (coarse -> fine)
            dt_list = [2e-2, 1e-2, 5e-3, 2.5e-3, 1.25e-3]

        res_trap = self.temporal_convergence(
            method="trapezoid",
            dt_list=dt_list,
            Tfinal=Tfinal,
            dt_ref_factor=dt_ref_factor,
            expected_order=2
        )

        res_be = self.temporal_convergence(
            method="backward_euler",
            dt_list=dt_list,
            Tfinal=Tfinal,
            dt_ref_factor=dt_ref_factor,
            expected_order=1
        )

        return res_trap, res_be

    # -------------------------
    # Spatial convergence study (optional but often expected)
    # -------------------------
    def spatial_convergence(
        self,
        method: str,
        m_list,
        dt_fixed: float,
        Tfinal: float,
        m_ref: int | None = None,
        expected_order: int = 2,
        make_plot: bool = True,
        title: str | None = None,
    ):
        """
        Spatial convergence test:
          - Fix dt small (dt_fixed) so time error is negligible.
          - Compare each coarse-grid solution to a fine-grid reference (m_ref),
            interpolated onto the coarse x grid.
          - Error measured at final time.

        Returns dict with dx list, error list, observed orders.
        """
        m_list = np.array(m_list, dtype=int)
        if m_ref is None:
            m_ref = int(np.max(m_list) * 2)

        # reference solution on fine grid
        x_ref, W_ref, beam_ref = self.solve_final_W(method=method, dt=dt_fixed, Tfinal=Tfinal, m_override=m_ref)

        dxs = []
        errs = []

        for m in m_list:
            x, W, beam_c = self.solve_final_W(method=method, dt=dt_fixed, Tfinal=Tfinal, m_override=int(m))
            dx = x[1] - x[0]
            dxs.append(dx)

            # interpolate reference onto coarse grid
            W_ref_on_coarse = np.interp(x, x_ref, W_ref)
            errs.append(self.rel_L2_discrete(x, W, W_ref_on_coarse))

        # estimate observed order
        dx_sorted, e_sorted, p_obs = self.estimate_order(dxs, errs)

        out = dict(
            method=method,
            dt_fixed=dt_fixed,
            m=np.array(m_list),
            dx=np.array(dx_sorted),
            err=np.array(e_sorted),
            p_obs=np.array(p_obs),
            expected_order=expected_order,
            m_ref=m_ref
        )

        if make_plot:
            plt.figure()
            plt.loglog(out["dx"], out["err"], "o-", label="Relative L2 error at T")

            ref = out["err"][-1] * (out["dx"] / out["dx"][-1]) ** expected_order
            plt.loglog(out["dx"], ref, "--", label=f"O(Δx^{expected_order}) reference")

            plt.xlabel("Δx")
            plt.ylabel("‖e‖ (relative discrete L2)")
            if title is None:
                plt.title(f"Spatial convergence: {method} (dt={dt_fixed}, T={Tfinal})")
            else:
                plt.title(title)
            plt.grid(True, which="both")
            plt.legend()
            plt.show()

            print(f"\nSpatial convergence ({method}):")
            print(f"  m_ref = {m_ref}, dt_fixed = {dt_fixed:.3e}")
            for i in range(len(out["p_obs"])):
                print(f"  p between dx={out['dx'][i]:.3e} and {out['dx'][i+1]:.3e} : {out['p_obs'][i]:.3f}")

        return out
