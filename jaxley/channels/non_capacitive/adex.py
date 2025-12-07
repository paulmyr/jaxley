
"""
Unified AdEx (Adaptive Exponential Integrate-and-Fire) neuron model.

References:
    Brette, R., & Gerstner, W. (2005). Adaptive exponential integrate-and-fire model
    as an effective description of neuronal activity.
"""

from typing import Optional
from warnings import warn

import jax
import jax.numpy as jnp
from jax import Array

from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler, save_exp

__all__ = ["AdEx", "AdExSurrogate"]

class AdEx(Channel):
    """
    Unified Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.

    This channel implements the full AdEx dynamics:
    - Leak current
    - Exponential spike mechanism
    - Adaptation current
    - Spike detection and reset

    The membrane equation is:
        C_m * dV/dt = -g_L * (V - E_L) + g_L * delta_T * exp((V - V_T) / delta_T) - w + I_ext
        tau_w * dw/dt = a * (V - E_L) - w

    When V > v_threshold:
        V -> v_reset
        w -> w + b

    Parameters (set via channel_params):
        g_L: Leak conductance (default: 10.0 nS)
        E_L: Leak reversal potential (default: -70.0 mV)
        v_T: Spike threshold potential (default: -50.0 mV)
        delta_T: Spike slope factor (default: 2.0 mV)
        v_threshold: Spike detection threshold (default: 20.0 mV)
        v_reset: Reset potential after spike (default: -70.0 mV)
        tau_w: Adaptation time constant (default: 30.0 ms)
        a: Subthreshold adaptation (default: 2.0 nS)
        b: Spike-triggered adaptation (default: 0.0 pA)
    """

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)

        prefix = self._name

        # All AdEx parameters
        # Note: Conductances are in S/cm^2, currents in mA/cm^2
        # These defaults give reasonable tonic spiking behavior
        self.channel_params = {
            # Leak parameters
            f"{prefix}_C_m": 200.,
            f"{prefix}_g_L": 6e-5,          # S/cm^2 (leak conductance density)
            f"{prefix}_E_L": -70.0,         # mV
            # Exponential spike parameters
            f"{prefix}_v_T": -50.0,         # mV (spike threshold)
            f"{prefix}_delta_T": 2.0,       # mV (spike slope factor)
            # Spike reset parameters
            f"{prefix}_v_threshold": 0.0,   # mV (detection threshold)
            f"{prefix}_v_reset": -70.0,     # mV (reset potential)
            # Adaptation parameters
            f"{prefix}_tau_w": 30.0,        # ms
            f"{prefix}_a": 1e-5,            # S/cm^2 (sub-threshold adaptation)
            f"{prefix}_b": 3e-7,            # mA/cm^2 (spike-triggered adaptation)
        }

        # AdEx state variables
        self.channel_states = {
            f"{prefix}_w": 0.0,      # Adaptation current
            f"{prefix}_spikes": False  # Spike indicator
        }

        self.current_name = f"i_{prefix}"

        warn(
            f"The {self.name} channel does not support surrogate gradients. "
            "Its gradient will be zero after every spike. "
            "Use AdExSurrogate for differentiable spiking."
        )

    def update_states(
        self,
        states: dict[str, Array],
        dt: float,
        v: Array,
        params: dict[str, Array]
    ) -> dict[str, Array]:
        """
        Update adaptation variable, integrate voltage, and handle spike reset.

        Like Izhikevich, all dynamics are handled here. This allows proper clamping
        of the exponential spike mechanism.

        Args:
            states: Current state dictionary
            dt: Time step (ms)
            v: Membrane potential (mV)
            params: Parameter dictionary

        Returns:
            Dictionary with updated states: v, w, spikes
        """
        prefix = self._name

        # Get parameters
        C_m = params[f"{prefix}_C_m"]
        g_L = params[f"{prefix}_g_L"]
        E_L = params[f"{prefix}_E_L"]
        v_T = params[f"{prefix}_v_T"]
        v_reset = params[f"{prefix}_v_reset"]
        v_threshold = params[f"{prefix}_v_threshold"]
        delta_T = params[f"{prefix}_delta_T"]

        tau_w = params[f"{prefix}_tau_w"]
        a = params[f"{prefix}_a"]
        b = params[f"{prefix}_b"]

        # Get current adaptation state
        w = states[f"{prefix}_w"]

        # Update adaptation variable with exponential Euler
        # dw/dt = (a * (v - E_L) - w) / tau_w
        w = exponential_euler(w, dt, a * (v - E_L), tau_w)

        # Update voltage with Forward Euler (nonlinear, like Izhikevich)
        # C * dv/dt = -g_L*(v-E_L) + g_L*delta_T*exp((v-v_T)/delta_T) - w
        # Note: External current will be added by Jaxley's voltage solver on top of this... I hope

        # Leak current
        i_leak = g_L * (v - E_L)

        # Exponential spike current (clamped to prevent overflow)
        exp_arg = (v - v_T) / delta_T
        exp_arg = jnp.minimum(exp_arg, 10.0)  # Clamp at exp(10) ≈ 22000
        i_exp = g_L * delta_T * save_exp(exp_arg)

        # Adaptation current
        i_adapt = w

        # Total derivative
        dv = (-i_leak + i_exp - i_adapt) / C_m
        v = v + dt * dv

        # Check for spike and reset
        condition = v >= v_threshold
        v = jax.lax.select(condition, v_reset, v)
        w = jax.lax.select(condition, w + b, w)

        return {
            "v": v,
            f"{prefix}_w": w,
            f"{prefix}_spikes": condition.astype(jnp.float32)
        }

    def compute_current(
                self,
                states: dict[str, Array],
                v: Array,
                params: dict[str, Array]
        ) -> Array:
        """
        Return zero current since all dynamics are handled in update_states.

        Like Izhikevich, AdEx integrates voltage directly in update_states,
        so compute_current returns zero to avoid double-integration.
        """
        return jnp.zeros((1,))


    def init_state(
        self,
        states: dict[str, Array],
        v: Array,
        params: dict[str, Array],
        delta_t: float
    ) -> dict[str, Array]:
        """
        """
        return {}


class AdExSurrogate(AdEx):
    """
    Adaptive Exponential Integrate-and-Fire with surrogate gradients.

    This is the differentiable version of AdEx that uses surrogate gradients
    to enable gradient-based optimization through spike events.

    The forward pass uses hard thresholds (same behavior as AdEx), but the
    backward pass uses smooth surrogate gradients for the spike function.

    Args:
        surrogate_type: Type of surrogate gradient ('sigmoid', 'exponential', 'superspike')
        surrogate_slope: Steepness parameter for the surrogate gradient (default: 10.0)
        name: Optional name for the channel

    Example:
        >>> cell = jx.Cell()
        >>> cell.insert(AdExSurrogate(surrogate_type="sigmoid", surrogate_slope=10.0))
        >>> cell.make_trainable("AdExSurrogate_g_L")
    """

    def __init__(
        self,
        surrogate_type: str = "sigmoid",
        surrogate_slope: float = 10.0,
        name: Optional[str] = None
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__("AdEx")

        # Store surrogate gradient configuration
        self.surrogate_type = surrogate_type
        self.surrogate_slope = surrogate_slope

        # Select surrogate function
        if surrogate_type == "sigmoid":
            self.surrogate_fn = lambda x: _heaviside_with_sigmoid_surrogate(
                x, slope=surrogate_slope
            )
        elif surrogate_type == "exponential":
            self.surrogate_fn = lambda x: _heaviside_with_exponential_surrogate(
                x, beta=surrogate_slope
            )
        elif surrogate_type == "superspike":
            self.surrogate_fn = lambda x: _heaviside_with_superspike_surrogate(
                x, beta=surrogate_slope
            )
        else:
            raise ValueError(
                f"Unknown surrogate type: {surrogate_type}. "
                "Choose from: 'sigmoid', 'exponential', 'superspike'"
            )

    def update_states(
        self,
        states: dict[str, Array],
        dt: float,
        v: Array,
        params: dict[str, Array]
    ) -> dict[str, Array]:
        """
        Update adaptation variable and handle spike reset with surrogate gradients.

        Forward pass: Hard threshold (same as AdEx)
        Backward pass: Smooth surrogate gradient
        """
        prefix = self._name

        # Get parameters
        a = params[f"{prefix}_a"]
        E_L = params[f"{prefix}_E_L"]
        tau_w = params[f"{prefix}_tau_w"]
        b = params[f"{prefix}_b"]
        v_threshold = params[f"{prefix}_v_threshold"]
        v_reset = params[f"{prefix}_v_reset"]

        # Get current adaptation state
        w = states[f"{prefix}_w"]

        # Update adaptation variable
        w_new = exponential_euler(w, dt, a * (v - E_L), tau_w)

        # Check for spike with surrogate gradient (differentiable)
        spike_occurred = self.surrogate_fn(v - v_threshold)

        # Reset voltage and update adaptation (differentiable)
        v_new = spike_occurred * v_reset + (1.0 - spike_occurred) * v
        w_new = spike_occurred * (w_new + b) + (1.0 - spike_occurred) * w_new

        return {
            "v": v_new,
            f"{prefix}_w": w_new,
            f"{prefix}_spikes": spike_occurred
        }


# ============================================================================
# Surrogate Gradient Functions
# ============================================================================

def _heaviside_with_sigmoid_surrogate(x: Array, slope: float = 10.0) -> Array:
    """
    Heaviside step function with sigmoid surrogate gradient.

    Forward pass: H(x) = 1 if x >= 0 else 0
    Backward pass: sigmoid'(x) = slope * sigmoid(x) * (1 - sigmoid(x))

    Args:
        x: Input (typically v - v_threshold)
        slope: Steepness of the surrogate gradient

    Returns:
        Step function output with smooth gradient
    """
    @jax.custom_vjp                 # https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html
    def heaviside_surrogate(x):
        return (x >= 0.0).astype(jnp.float32)

    def heaviside_fwd(x):
        return heaviside_surrogate(x), x

    def heaviside_bwd(x, g):
        sigmoid_x = jax.nn.sigmoid(slope * x)
        grad_surrogate = slope * sigmoid_x * (1 - sigmoid_x)
        return (g * grad_surrogate,)

    heaviside_surrogate.defvjp(heaviside_fwd, heaviside_bwd)
    return heaviside_surrogate(x)


def _heaviside_with_exponential_surrogate(x: Array, beta: float = 5.0) -> Array:
    """
    Heaviside step function with exponential surrogate gradient.

    This is popular in spiking neural networks literature.
    Surrogate: grad = beta * exp(-beta * |x|)

    Args:
        x: Input (typically v - v_threshold)
        beta: Width parameter

    Returns:
        Step function output with smooth gradient
    """
    @jax.custom_vjp                 # https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html
    def heaviside_surrogate(x):
        return (x >= 0.0).astype(jnp.float32)

    def heaviside_fwd(x):
        return heaviside_surrogate(x), x

    def heaviside_bwd(x, g):
        grad_surrogate = beta * jnp.exp(-beta * jnp.abs(x))
        return (g * grad_surrogate,)

    heaviside_surrogate.defvjp(heaviside_fwd, heaviside_bwd)
    return heaviside_surrogate(x)


def _heaviside_with_superspike_surrogate(x: Array, beta: float = 10.0) -> Array:
    """
    Heaviside with SuperSpike surrogate (Zenke & Ganguli, 2018).

    SuperSpike: grad = 1 / (beta * |x| + 1)^2

    Args:
        x: Input (typically v - v_threshold)
        beta: Sharpness parameter

    Returns:
        Step function output with smooth gradient
    """
    @jax.custom_vjp                 # https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html
    def heaviside_surrogate(x):
        return (x >= 0.0).astype(jnp.float32)

    def heaviside_fwd(x):
        return heaviside_surrogate(x), x

    def heaviside_bwd(x, g):
        grad_surrogate = 1.0 / jnp.square(beta * jnp.abs(x) + 1.0)
        return (g * grad_surrogate,)

    heaviside_surrogate.defvjp(heaviside_fwd, heaviside_bwd)
    return heaviside_surrogate(x)
