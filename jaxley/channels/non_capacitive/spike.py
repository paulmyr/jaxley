# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional
from warnings import warn

import jax
import jax.numpy as jnp

from jaxley.channels import Channel

__all__ = ["Fire", "FireSurrogate"]

class Fire(Channel):
    """Mechanism to reset the voltage when it crosses a threshold.

    When combined with a ``Leak`` channel, this can be used to implement
    leaky-integrate-and-fire neurons.

    Note that, after the voltage is reset by this channel, other channels (or external
    currents), can still modify the membrane voltage `within the same time step`.
    """

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {f"{self.name}_vth": -50, f"{self.name}_vreset": -70}
        self.channel_states = {f"{self.name}_spikes": False}
        self.current_name = f"{self.name}_fire"
        warn(
            "The `Fire` channel does not support surrogate gradients. Its gradient "
            "will be zero after every spike."
        )

    def update_states(self, states, dt, v, params):
        """Reset the voltage when a spike occurs and log the spike"""
        prefix = self._name
        vreset = params[f"{prefix}_vreset"]
        vth = params[f"{prefix}_vth"]

        spike_occurred = v > vth
        v = jax.lax.select(spike_occurred, vreset, v)
        return {"v": v, f"{self.name}_spikes": spike_occurred}

    def compute_current(self, states, v, params):
        return jnp.zeros((1,))

    def init_state(self, states, v, params, delta_t):
        return {}



class FireSurrogate(Channel):
    """
    Fire mechanism with surrogate gradients for differentiable LIF neurons.

    This channel resets the voltage when it crosses a threshold, but uses
    a smooth surrogate gradient during backpropagation to enable gradient-based
    optimization.

    Args:
        surrogate_type: Type of surrogate gradient ('sigmoid', 'exponential', 'superspike')
        surrogate_slope: Steepness parameter for the surrogate gradient
        name: Optional name for the channel
    """

    def __init__(
        self,
        surrogate_type: str = "sigmoid",
        surrogate_slope: float = 10.0,
        name: Optional[str] = None
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)

        self.channel_params = {
            f"{self.name}_vth": -50.0,
            f"{self.name}_vreset": -70.0,
        }
        self.channel_states = {f"{self.name}_spikes": False}
        self.current_name = f"{self.name}_fire"

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

    def update_states(self, states, dt, v, params):
        """
        Reset the voltage when a spike occurs using surrogate gradients.

        Forward pass: Hard threshold with reset
        Backward pass: Smooth surrogate gradient
        """
        prefix = self._name
        vreset = params[f"{prefix}_vreset"]
        vth = params[f"{prefix}_vth"]

        # Compute spike with surrogate gradient
        # spike_occurred is differentiable w.r.t. v and vth!
        spike_occurred = self.surrogate_fn(v - vth)

        # Reset voltage: v_new = spike * vreset + (1 - spike) * v
        # This is differentiable!
        v_new = spike_occurred * vreset + (1.0 - spike_occurred) * v

        return {"v": v_new, f"{self.name}_spikes": spike_occurred}

    def compute_current(self, states, v, params):
        """No current injection (reset happens in update_states)"""
        return jnp.zeros((1,))

    def init_state(self, states, v, params, delta_t):
        """No state initialization needed"""
        return {}


def _heaviside_with_sigmoid_surrogate(x, slope=10.0):
    """
    Heaviside step function with sigmoid surrogate gradient.

    Forward pass: H(x) = 1 if x >= 0 else 0
    Backward pass: sigmoid'(x) = slope * sigmoid(x) * (1 - sigmoid(x))

    Args:
        x: Input (typically v - vth)
        slope: Steepness of the surrogate gradient (higher = steeper)

    Returns:
        Step function output with smooth gradient
    """
    @jax.custom_vjp
    def heaviside_surrogate(x):
        # Forward pass: use real step function
        return (x >= 0.0).astype(jnp.float32)

    def heaviside_fwd(x):
        # Forward pass: return output and residuals for backward
        return heaviside_surrogate(x), x

    def heaviside_bwd(x, g):
        # Backward pass: use sigmoid derivative as surrogate
        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        sigmoid_x = jax.nn.sigmoid(slope * x)
        grad_surrogate = slope * sigmoid_x * (1 - sigmoid_x)
        return (g * grad_surrogate,)

    heaviside_surrogate.defvjp(heaviside_fwd, heaviside_bwd)
    return heaviside_surrogate(x)


def _heaviside_with_exponential_surrogate(x, beta=5.0):
    """
    Heaviside step function with exponential surrogate gradient.

    This is a popular choice in spiking neural networks literature.
    The surrogate is: grad = beta * exp(-beta * |x|)

    Args:
        x: Input (typically v - vth)
        beta: Width parameter (higher = narrower spike)

    Returns:
        Step function output with smooth gradient
    """
    @jax.custom_vjp
    def heaviside_surrogate(x):
        # Forward pass: use real step function
        return (x >= 0.0).astype(jnp.float32)

    def heaviside_fwd(x):
        return heaviside_surrogate(x), x

    def heaviside_bwd(x, g):
        # Backward pass: use exponential surrogate
        # grad = beta * exp(-beta * |x|)
        grad_surrogate = beta * jnp.exp(-beta * jnp.abs(x))
        return (g * grad_surrogate,)

    heaviside_surrogate.defvjp(heaviside_fwd, heaviside_bwd)
    return heaviside_surrogate(x)


def _heaviside_with_superspike_surrogate(x, beta=10.0):
    """
    Heaviside with SuperSpike surrogate (Zenke & Ganguli, 2018).

    SuperSpike: grad = 1 / (beta * |x| + 1)^2

    This has been shown to work well for training spiking networks.

    Args:
        x: Input (typically v - vth)
        beta: Sharpness parameter

    Returns:
        Step function output with smooth gradient
    """
    @jax.custom_vjp
    def heaviside_surrogate(x):
        # Forward pass: use real step function
        return (x >= 0.0).astype(jnp.float32)

    def heaviside_fwd(x):
        return heaviside_surrogate(x), x

    def heaviside_bwd(x, g):
        # Backward pass: use SuperSpike surrogate
        # grad = 1 / (beta * |x| + 1)^2
        grad_surrogate = 1.0 / jnp.square(beta * jnp.abs(x) + 1.0)
        return (g * grad_surrogate,)

    heaviside_surrogate.defvjp(heaviside_fwd, heaviside_bwd)
    return heaviside_surrogate(x)
