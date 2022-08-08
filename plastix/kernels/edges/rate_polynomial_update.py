from plastix.kernels import State, Parameter
from plastix.kernels.edges import EdgeKernel
import jax.numpy as jnp


def _get_coefficient_pow(pre, post, weight):
    coefficient_pow = jnp.outer(
        jnp.outer(
            jnp.array([pre**0, pre**1, pre**2]),
            jnp.array([post**0, post**1, post**2]),
        ),
        jnp.array([weight**0, weight**1, weight**2]),
    )
    coefficient_pow = jnp.reshape(coefficient_pow, (3, 3, 3))
    return coefficient_pow


class RatePolynomialUpdate(EdgeKernel):

    signal = State((1,), jnp.zeros)
    weight = Parameter((1,), jnp.ones)
    lr = Parameter((1,), jnp.ones, shared=True)
    # create meta variable A, that assigns weight to each polynomial
    # coefficient of the update rule
    A = Parameter((3, 3, 3), jnp.ones, shared=True)

    def update_state(self, input_node):
        self.signal = input_node.rate * self.weight

    def update_parameters(self, input_node, output_node):
        coefficient_pow = _get_coefficient_pow(
            input_node.rate, output_node.rate, self.weight
        )
        dw = jnp.sum(jnp.multiply(self.A, coefficient_pow))
        self.weight += self.lr * dw
