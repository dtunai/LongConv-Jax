import jax
import time
import numpy as np
import jax.numpy as jnp
from longconv_jax.kernels.long_convs import LongConvModel

# Init constants
D_INPUT = 3
D_OUTPUT = 10
BATCH_SIZE = 5
SEQ_LENGTH = 1024
D_MODEL = 512
N_LAYERS = 4
DROPOUT = 0.1
PRENORM = True
LONG_CONV_ARGS = {
    "kernel_dropout_rate": 0.2,
    "kernel_learning_rate": 0.001,
    "kernel_lam": 0.001,
}

# Dummy data
dummy_data_np = np.random.randn(BATCH_SIZE, SEQ_LENGTH, D_INPUT)
dummy_data_jax = jnp.array(dummy_data_np)

# Initialize model
model = LongConvModel(
    d_input=D_INPUT,
    d_output=D_OUTPUT,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    prenorm=PRENORM,
    conv_kwargs=LONG_CONV_ARGS,
)
# Init params
params = model.init(jax.random.PRNGKey(0), dummy_data_jax)


# Apply @jax.jit
@jax.jit
def forward(params, x):
    return model.apply(params, x, deterministic=True)


# Execute forward pass
start_time = time.time()
output = forward(params, dummy_data_jax)
end_time = time.time()

# Output results
print("Execution time (FW): {:.4f}/s".format(end_time - start_time))
print("Output shape:", output.shape)
