import math

import torch


class Layer:
    def _init_weights(self, size, weights_init, device):
        # Initialize weights using He initialization
        n_inputs = size[0]
        if weights_init == "normal":
            std = math.sqrt(2.0 / n_inputs)
            return torch.normal(mean=0, std=std, size=size, device=device)
        else:
            limit = math.sqrt(6.0 / n_inputs)
            return torch.empty(size, device=device).uniform_(-limit, limit)

    def _init_biases(self, size, device):
        return torch.zeros(size, device=device)


class Input(Layer):
    def __init__(self, size):
        self.type = "input"
        self.size = size
        self.a = None

    def initialize(self, device):
        self.a = torch.zeros((self.size, 1), device=device)
        return self


class Input3D(Layer):
    def __init__(self, size):
        self.type = "input_3d"
        self.size = size
        self.a = None

    def initialize(self, device):
        self.a = torch.zeros((self.size[0], self.size[1], self.size[2]), device=device)
        return self


class FullyConnected(Layer):
    def __init__(self, size, activation, name=None):
        self.type = "fully_connected"
        self.name = name
        self.learnable = True
        self.size = size
        self.activation = activation
        self.device = None
        self.z = None
        self.a = None
        self.w = None
        self.b = None
        self.grad_w = None
        self.grad_b = None

    def _calculate_layer_error(self, next_layer_error, next_layer):
        if not next_layer:
            layer_error = next_layer_error * self.activation.derivative(self.z)
        else:
            layer_error = torch.matmul(
                next_layer.w.t(), next_layer_error
            ) * self.activation.derivative(self.z)

        return layer_error

    def initialize(self, previous_layer, weights_init, device):
        self.device = device
        self.z = torch.zeros((self.size, 1), device=self.device)
        self.a = torch.zeros((self.size, 1), device=self.device)
        self.w = super()._init_weights(
            (self.size, previous_layer.size), weights_init, self.device
        )
        self.b = super()._init_biases((self.size, 1), self.device)
        self.grad_w = torch.zeros_like(self.w, device=self.device)
        self.grad_b = torch.zeros_like(self.b, device=self.device)

        return self

    def zero_grad(self):
        self.grad_w = torch.zeros_like(self.w)
        self.grad_b = torch.zeros_like(self.b)
        return

    def forward(self, input_data):
        self.z = torch.matmul(self.w, input_data) + self.b
        self.a = self.activation.apply(self.z)
        return

    def backward(self, next_layer_error, prev_layer, next_layer=None):
        layer_error = self._calculate_layer_error(next_layer_error, next_layer)

        self.grad_w += torch.matmul(layer_error, prev_layer.a.t())
        self.grad_b += layer_error

        return layer_error


class Convolutional(Layer):
    def __init__(
        self,
        activation,
        filters_num=4,
        kernel_size=3,
        padding_type=None,
        compute_mode="fast",
        name=None,
    ):
        if compute_mode not in ("ordinary", "fast"):
            raise Exception(f"Invalid compute_mode: {compute_mode}")

        if padding_type not in (None, "same"):
            raise Exception(f"Invalid padding_type: {padding_type}")

        # 'same' padding requires an odd kernel size for symmetry
        if padding_type == "same" and kernel_size % 2 == 0:
            raise Exception(
                'padding_type="same" should be used only with odd kernel_size'
            )

        self.type = "convolutional"
        self.name = name
        self.learnable = True
        self.activation = activation
        self.filters_num = filters_num
        self.kernel_size = kernel_size
        self.padding_type = padding_type
        self.compute_mode = compute_mode
        self.padding = (
            int((self.kernel_size - 1) / 2) if self.padding_type else 0
        )  # Padding size for 'same' padding
        self.device = None
        self.input_c = None
        self.input_h = None
        self.input_w = None
        self.output_c = None
        self.output_h = None
        self.output_w = None
        self.z = None
        self.a = None
        self.w_shape = None
        self.w = None
        self.b = None
        self.grad_w = None
        self.grad_b = None

    def _init_weights(self, size, weights_init, device):
        # Initialize weights using He initialization
        n_inputs = size[1] * size[2] * size[3]
        if weights_init == "normal":
            std = math.sqrt(2.0 / n_inputs)
            return torch.normal(mean=0, std=std, size=size, device=device)
        else:
            limit = math.sqrt(6.0 / n_inputs)
            return torch.empty(size, device=device).uniform_(-limit, limit)

    def _get_padded_tensor(self, tensor, padding, padding_value=0.0):
        p_top, p_bot, p_left, p_right = padding
        f, h, w = tensor.shape[0], tensor.shape[1], tensor.shape[2]

        padded_h = h + p_top + p_bot
        padded_w = w + p_left + p_right

        # Create new padded tensor and place the original tensor in it
        padded_tensor = torch.full(
            (f, padded_h, padded_w), padding_value, device=self.device
        )
        padded_tensor[:, p_top : p_top + h, p_left : p_left + w] = tensor[:].clone()

        return padded_tensor

    def _calculate_layer_error(self, next_layer_error, next_layer):
        if next_layer.type == "convolutional":
            k_next = next_layer.kernel_size

            # Determine the padding list based on the next layer's padding type
            if next_layer.padding_type == "same":
                padding_list = [int((k_next - 1) / 2)] * 4
            else:
                padding_list = [k_next - 1] * 4

            # Pad the error from the next layer
            next_layer_error_with_padding = self._get_padded_tensor(
                next_layer_error, padding_list
            )
            # Flip the next layer's weights for deconvolution
            flipped_w_next = torch.flip(next_layer.w, (2, 3))

            # Perform deconvolution to calculate the error for this layer
            layer_error = self._deconvolution(
                next_layer_error_with_padding,
                flipped_w_next,
                next_layer.output_c,
                k_next,
            )

            layer_error *= self.activation.derivative(self.z)
        else:
            layer_error = next_layer_error

        return layer_error

    def _fast_deconvolution(
        self, next_layer_error, filters, next_output_c, next_kernel_size
    ):
        # Unfold the error tensor into sliding window blocks
        unfolded_regions = next_layer_error.unfold(1, next_kernel_size, 1).unfold(
            2, next_kernel_size, 1
        )
        unfolded_regions = unfolded_regions.contiguous().view(
            next_output_c, self.output_h * self.output_w, -1
        )

        # Reshape filters for matrix multiplication
        reshaped_filters = filters.view(
            next_output_c, self.output_c, next_kernel_size * next_kernel_size
        )

        # Perform matrix multiplication using Einstein summation convention
        result = torch.einsum("abc,adc->db", unfolded_regions, reshaped_filters)

        return result.view(self.output_c, self.output_h, self.output_w)

    def _ordinary_deconvolution(self, next_layer_error, filters, next_kernel_size):
        layer_error = torch.zeros(
            (self.output_c, self.output_h, self.output_w), device=self.device
        )
        # Perform deconvolution by iterating over each filter and position
        for f in range(self.output_c):
            for i in range(self.output_h):
                for j in range(self.output_w):
                    layer_error[f][i][j] = torch.sum(
                        next_layer_error[
                            :, i : i + next_kernel_size, j : j + next_kernel_size
                        ]
                        * filters[:, f]
                    )
        return layer_error

    def _deconvolution(
        self, next_layer_error, filters, next_output_c, next_kernel_size
    ):
        if self.compute_mode == "fast":
            return self._fast_deconvolution(
                next_layer_error, filters, next_output_c, next_kernel_size
            )
        else:
            return self._ordinary_deconvolution(
                next_layer_error, filters, next_kernel_size
            )

    # Fast convolution using Einstein summation convention
    def _fast_convolution(self, input_image):
        # Unfold the input tensor into sliding window blocks
        unfolded_regions = input_image.unfold(1, self.kernel_size, 1).unfold(
            2, self.kernel_size, 1
        )
        unfolded_regions = unfolded_regions.contiguous().view(
            self.input_c, self.output_h * self.output_w, -1
        )

        # Reshape the weights for matrix multiplication
        reshaped_filters = self.w.view(self.output_c, self.input_c, -1)

        # Perform convolution using matrix multiplication
        result = torch.einsum("abc,dac->db", unfolded_regions, reshaped_filters)
        result = result.view(self.output_c, self.output_h, self.output_w)

        result += self.b.view(-1, 1, 1)

        return result

    def _ordinary_convolution(self, input_image):
        z = torch.zeros_like(self.z)
        # Perform convolution by iterating over each filter and position
        for f in range(self.output_c):
            for i in range(self.output_h):
                for j in range(self.output_w):
                    z[f][i][j] = (
                        torch.sum(
                            input_image[
                                :, i : i + self.kernel_size, j : j + self.kernel_size
                            ]
                            * self.w[f]
                        )
                        + self.b[f]
                    )
        return z

    def _convolution(self, input_image):
        if self.padding_type == "same":
            # Apply padding if needed
            padding_list = [self.padding] * 4
            input_image = self._get_padded_tensor(input_image, padding_list)

        if self.compute_mode == "fast":
            return self._fast_convolution(input_image)
        else:
            return self._ordinary_convolution(input_image)

    # Fast gradient update during backpropagation using Einstein summation
    def _fast_update_gradients(self, layer_error, prev_layer, prev_layer_a):
        # Unfold previous layer activations into sliding window blocks
        unfolded_prev_layer_a = prev_layer_a.unfold(1, self.output_h, 1).unfold(
            2, self.output_w, 1
        )
        unfolded_prev_layer_a = unfolded_prev_layer_a.contiguous().view(
            self.input_c, self.kernel_size * self.kernel_size, -1
        )

        # Reshape layer error for efficient gradient computation
        reshaped_layer_error = layer_error.view(self.output_c, -1)

        # Update gradients using matrix multiplication
        grad_w_update = torch.einsum(
            "ab,cdb->acd", reshaped_layer_error, unfolded_prev_layer_a
        )
        grad_b_update = torch.einsum("f...->f", reshaped_layer_error)

        self.grad_w += grad_w_update.view(
            self.output_c, self.input_c, self.kernel_size, self.kernel_size
        )
        self.grad_b += grad_b_update.view(self.output_c)

        return

    def _ordinary_update_gradients(self, layer_error, prev_layer, prev_layer_a):
        # Update gradients for weights and biases using nested loops
        for f in range(self.output_c):
            layer_error_f = layer_error[f]
            for c in range(self.input_c):
                prev_layer_a_c = prev_layer_a[c]
                for m in range(self.kernel_size):
                    for n in range(self.kernel_size):
                        # Compute weight gradient for each kernel position
                        self.grad_w[f][c][m][n] += torch.sum(
                            layer_error_f
                            * prev_layer_a_c[
                                m : m + self.output_h, n : n + self.output_w
                            ]
                        )
            # Compute bias gradient
            self.grad_b[f] += torch.sum(layer_error_f)
        return

    def _update_gradients(self, layer_error, prev_layer):
        # Choose the gradient update method based on compute mode
        prev_layer_a = prev_layer.a
        if self.padding_type == "same":
            padding_list = [self.padding] * 4
            prev_layer_a = self._get_padded_tensor(prev_layer.a, padding_list)

        if self.compute_mode == "fast":
            return self._fast_update_gradients(layer_error, prev_layer, prev_layer_a)
        else:
            return self._ordinary_update_gradients(
                layer_error, prev_layer, prev_layer_a
            )

    def initialize(self, previous_layer, weights_init, device):
        self.device = device
        if previous_layer.type in ("convolutional", "max_pool_2d"):
            self.input_c = previous_layer.output_c
            self.input_h = previous_layer.output_h
            self.input_w = previous_layer.output_w
        else:
            self.input_c = previous_layer.size[0]
            self.input_h = previous_layer.size[1]
            self.input_w = previous_layer.size[2]

        self.output_c = self.filters_num
        self.output_h = (self.input_h - self.kernel_size + 2 * self.padding) + 1
        self.output_w = (self.input_w - self.kernel_size + 2 * self.padding) + 1

        self.z = torch.zeros(
            (self.output_c, self.output_h, self.output_w), device=self.device
        )
        self.a = torch.zeros(
            (self.output_c, self.output_h, self.output_w), device=self.device
        )
        self.w_shape = (self.output_c, self.input_c, self.kernel_size, self.kernel_size)
        self.w = self._init_weights(self.w_shape, weights_init, self.device)
        self.b = super()._init_biases(self.output_c, self.device)
        self.grad_w = torch.zeros_like(self.w)
        self.grad_b = torch.zeros_like(self.b)

        return self

    def zero_grad(self):
        self.grad_w = torch.zeros_like(self.w)
        self.grad_b = torch.zeros_like(self.b)
        return

    def forward(self, input_image):
        self.z = self._convolution(input_image)
        self.a = self.activation.apply(self.z)
        return

    def backward(self, next_layer_error, prev_layer, next_layer):
        layer_error = self._calculate_layer_error(next_layer_error, next_layer)
        self._update_gradients(layer_error, prev_layer)
        return layer_error


class Flatten:
    def __init__(self, name=None):
        self.type = "flatten"
        self.name = name
        self.learnable = False
        self.size = None
        self.a = None

    def initialize(self, previous_layer, device):
        if previous_layer.type not in ("convolutional", "max_pool_2d"):
            raise Exception(
                "Flatten layer should be used only after convolutional or max pool"
            )

        self.size = (
            previous_layer.output_c * previous_layer.output_h * previous_layer.output_w
        )
        self.a = torch.zeros((self.size, 1), device=device)

        return self

    def forward(self, input_data):
        self.a = input_data.flatten()
        self.a = self.a.reshape(self.a.size(0), 1)
        return

    def backward(self, next_layer_error, prev_layer, next_layer):
        reshape_sizes = (prev_layer.output_c, prev_layer.output_h, prev_layer.output_w)
        layer_error = (
            torch.matmul(next_layer.w.t(), next_layer_error) * torch.ones_like(self.a)
        ).reshape(reshape_sizes)

        return layer_error


class MaxPool2D:
    def __init__(self, name=None, compute_mode="fast"):
        if compute_mode not in ("ordinary", "fast"):
            raise Exception(f"Invalid compute_mode: {compute_mode}")

        self.type = "max_pool_2d"
        self.name = name
        self.compute_mode = compute_mode
        self.learnable = False
        self.device = None
        self.input_h = None
        self.input_w = None
        self.input_c = None
        self.output_h = None
        self.output_w = None
        self.output_c = None
        self.a = None
        self.max_indices = None

    def initialize(self, previous_layer, device):
        if previous_layer.type != "convolutional":
            raise Exception("Max pool layer should be used only after convolutional")

        self.device = device
        self.input_h = previous_layer.output_h
        self.input_w = previous_layer.output_w
        self.input_c = previous_layer.output_c
        self.output_h = int(self.input_h / 2)  # Assuming a 2x2 pool size
        self.output_w = int(self.input_w / 2)
        self.output_c = self.input_c
        self.a = torch.zeros(
            (self.output_c, self.output_h, self.output_w), device=device
        )
        self.max_indices = torch.zeros(
            (self.output_c, self.output_h, self.output_w, 2), device=device
        )

        return self

    def _fast_forward(self, input_data):
        unfolded_input_data = input_data.unfold(1, 2, 2).unfold(2, 2, 2)

        # Reshape and compute max values and their indices
        self.a, indices = unfolded_input_data.reshape(
            self.output_c, self.output_h, self.output_w, -1
        ).max(dim=-1)

        # Compute indices of max values
        y_indices = indices // 2
        x_indices = indices % 2

        self.max_indices = torch.stack((y_indices, x_indices), dim=-1)

        return

    def _ordinary_forward(self, input_data):
        for f in range(self.output_c):
            input_data_f = input_data[f]
            for i in range(self.output_h):
                for j in range(self.output_w):
                    # Define the region of the input data to consider
                    i_start = i * 2
                    i_end = i * 2 + 2
                    j_start = j * 2
                    j_end = j * 2 + 2
                    region = input_data_f[i_start:i_end, j_start:j_end]

                    # Compute max value and its indices
                    self.a[f][i][j] = torch.max(region)
                    self.max_indices[f][i][j] = (region == self.a[f][i][j]).nonzero()[0]
        return self.a

    def _get_padded_tensor(self, tensor, padding, padding_value=0.0):
        p_top, p_bot, p_left, p_right = padding
        f, h, w = tensor.shape[0], tensor.shape[1], tensor.shape[2]

        padded_h = h + p_top + p_bot
        padded_w = w + p_left + p_right

        # Create new padded tensor and place the original tensor in it
        padded_tensor = torch.full(
            (f, padded_h, padded_w), padding_value, device=self.device
        )
        padded_tensor[:, p_top : p_top + h, p_left : p_left + w] = tensor[:].clone()

        return padded_tensor

    def _calculate_layer_error(self, next_layer_error, next_layer):
        k_next = next_layer.kernel_size

        # Determine the padding list based on the next layer's padding type
        if next_layer.padding_type == "same":
            padding_list = [int((k_next - 1) / 2)] * 4
        else:
            padding_list = [k_next - 1] * 4

        # Pad the error from the next layer
        next_layer_error_with_padding = self._get_padded_tensor(
            next_layer_error, padding_list
        )
        # Flip the next layer's weights for deconvolution
        flipped_w_next = torch.flip(next_layer.w, (2, 3))

        # Perform deconvolution to calculate the error for this layer
        layer_error = self._deconvolution(
            next_layer_error_with_padding,
            flipped_w_next,
            next_layer.output_c,
            k_next,
        )

        return layer_error

    def _fast_deconvolution(
        self, next_layer_error, filters, next_output_c, next_kernel_size
    ):
        # Unfold the error tensor into sliding window blocks
        unfolded_regions = next_layer_error.unfold(1, next_kernel_size, 1).unfold(
            2, next_kernel_size, 1
        )
        unfolded_regions = unfolded_regions.contiguous().view(
            next_output_c, self.output_h * self.output_w, -1
        )

        # Reshape filters for matrix multiplication
        reshaped_filters = filters.view(
            next_output_c, self.output_c, next_kernel_size * next_kernel_size
        )

        # Perform matrix multiplication using Einstein summation convention
        result = torch.einsum("abc,adc->db", unfolded_regions, reshaped_filters)

        return result.view(self.output_c, self.output_h, self.output_w)

    def _ordinary_deconvolution(self, next_layer_error, filters, next_kernel_size):
        layer_error = torch.zeros(
            (self.output_c, self.output_h, self.output_w), device=self.device
        )
        # Perform deconvolution by iterating over each filter and position
        for f in range(self.output_c):
            for i in range(self.output_h):
                for j in range(self.output_w):
                    layer_error[f][i][j] = torch.sum(
                        next_layer_error[
                            :, i : i + next_kernel_size, j : j + next_kernel_size
                        ]
                        * filters[:, f]
                    )
        return layer_error

    def _deconvolution(
        self, next_layer_error, filters, next_output_c, next_kernel_size
    ):
        if self.compute_mode == "fast":
            return self._fast_deconvolution(
                next_layer_error, filters, next_output_c, next_kernel_size
            )
        else:
            return self._ordinary_deconvolution(
                next_layer_error, filters, next_kernel_size
            )

    def _fast_layer_error_reconstruct_dimension(self, layer_error):
        layer_error_reconstruct = torch.zeros(
            (self.input_c, self.input_h, self.input_w), device=self.device
        )

        # Compute indices for updating layer_error based on max_indices
        filter_indices = (
            torch.arange(self.output_c, device=self.device)
            .view(-1, 1, 1)
            .expand(-1, self.output_h, self.output_w)
        )
        h_indices = (
            torch.arange(self.output_h, device=self.device)
            .view(1, -1, 1)
            .expand(self.output_c, -1, self.output_w)
        )
        w_indices = (
            torch.arange(self.output_w, device=self.device)
            .view(1, 1, -1)
            .expand(self.output_c, self.output_h, -1)
        )

        i_to_update = h_indices * 2 + self.max_indices[:, :, :, 0]
        j_to_update = w_indices * 2 + self.max_indices[:, :, :, 1]

        # Assign gradients to the appropriate locations
        layer_error_reconstruct[filter_indices, i_to_update, j_to_update] = layer_error

        return layer_error_reconstruct

    def _ordinary_layer_error_reconstruct_dimension(self, layer_error):
        layer_error_reconstruct = torch.zeros(
            (self.input_c, self.input_h, self.input_w), device=self.device
        )
        for f in range(self.output_c):
            layer_error_f = layer_error[f]
            for i in range(self.output_h):
                for j in range(self.output_w):
                    # Retrieve indices of the max value and update gradients
                    max_index_i, max_index_j = self.max_indices[f][i][j]
                    i_to_update = i * 2 + int(max_index_i)
                    j_to_update = j * 2 + int(max_index_j)
                    layer_error_reconstruct[f][i_to_update][j_to_update] = (
                        layer_error_f[i][j]
                    )
        return layer_error_reconstruct

    def _layer_error_reconstruct_dimension(self, next_layer_error):
        if self.compute_mode == "fast":
            return self._fast_layer_error_reconstruct_dimension(next_layer_error)
        else:
            return self._ordinary_layer_error_reconstruct_dimension(next_layer_error)

    def forward(self, input_data):
        if self.compute_mode == "fast":
            return self._fast_forward(input_data)
        else:
            return self._ordinary_forward(input_data)

    def backward(self, next_layer_error, prev_layer, next_layer):
        """
        If the next layer is convolutional, we first need to perform the deconvolution
        operation and only then execute the operation inverse to the one performed by
        the max pool layer (I called the inverse operation reconstruction).
        Finally, if deconvolution was applied, we need to apply the derivative of
        the activation function of the previous (before the max pool layer) convolutional layer.
        """
        if next_layer.type == "convolutional":
            layer_error = self._calculate_layer_error(next_layer_error, next_layer)
        else:
            layer_error = next_layer_error

        layer_error = self._layer_error_reconstruct_dimension(layer_error)

        if next_layer.type == "convolutional":
            layer_error *= prev_layer.activation.derivative(prev_layer.z)

        return layer_error
