def transpose_tensor(tensor, axes):
    if len(axes) != len(tensor.shape):
        raise ValueError("Invalid axes permutation")

    transposed_shape = [tensor.shape[axis] for axis in axes]
    transposed_tensor = create_empty_tensor(transposed_shape)

    indices = create_indices(transposed_shape)
    transposed_indices = create_transposed_indices(indices, axes)

    fill_transposed_tensor(tensor, transposed_tensor, transposed_indices)

    return transposed_tensor


def create_empty_tensor(shape):
    if len(shape) == 0:
        return []

    return [create_empty_tensor(shape[1:]) for _ in range(shape[0])]


def create_indices(shape):
    if len(shape) == 0:
        return []

    indices = [list(range(shape[0]))]
    remaining_indices = create_indices(shape[1:])

    for axis_indices in remaining_indices:
        expanded_axis_indices = [[i] * len(axis_indices) for i in range(shape[0])]
        axis_indices = [idx for sublist in expanded_axis_indices for idx in sublist]
        indices.append(axis_indices)

    return indices


def create_transposed_indices(indices, axes):
    transposed_indices = []
    for i, axis in enumerate(axes):
        transposed_indices.append(indices[i][axis])

    return transposed_indices


def fill_transposed_tensor(tensor, transposed_tensor, transposed_indices):
    if len(transposed_tensor) == 0:
        transposed_tensor[0] = tensor[tuple(transposed_indices)]
    else:
        for i in range(len(transposed_tensor)):
            fill_transposed_tensor(tensor, transposed_tensor[i], transposed_indices[i])

import numpy as np

tensor = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
axes = [1, 0, 2]
transposed_tensor = transpose_tensor(tensor, axes)
print(transposed_tensor)