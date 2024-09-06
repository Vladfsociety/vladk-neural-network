import math
import pprint
import random
import time

import torch


class Layer:
    def _init_weights(self, size, device):
        n_inputs = size[0]
        limit = math.sqrt(1 / n_inputs)
        return torch.tensor(
            [
                [random.uniform(-limit, limit) for _ in range(size[1])]
                for _ in range(size[0])
            ],
            device=device
        )

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
            layer_error = (torch.matmul(next_layer.w.t(), next_layer_error)
                           * self.activation.derivative(self.z))

        return layer_error

    def initialize(self, previous_layer, device):
        self.device = device
        self.z = torch.zeros((self.size, 1), device=self.device)
        self.a = torch.zeros((self.size, 1), device=self.device)
        self.w = super()._init_weights((self.size, previous_layer.size), self.device)
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
    def __init__(self, activation, filters_num=4, kernel_size=3, padding=0, stride=1, compute_mode="fast", name=None):
        self.type = "convolutional"
        self.name = name
        self.learnable = True
        self.activation = activation
        self.filters_num = filters_num
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.compute_mode = compute_mode
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

    def _init_weights(self, size, device):
        n_inputs = size[1] * size[2] * size[3]
        random.seed(41)

        limit = math.sqrt(1 / n_inputs)
        return torch.tensor(
            [
                [[[random.uniform(-limit, limit) for _ in range(size[3])]
                for _ in range(size[2])]
                for _ in range(size[1])]
                for _ in range(size[0])
            ],
            device=device
        )

    def _get_padded_tensor(self, tensor, padding, padding_value=0.0):
        p_top, p_bot, p_left, p_right = padding
        f, h, w = tensor.shape[0], tensor.shape[1], tensor.shape[2]

        padded_h = h + p_top + p_bot
        padded_w = w + p_left + p_right

        padded_tensor = torch.full((f, padded_h, padded_w), padding_value, device=self.device)
        padded_tensor[:, p_top:p_top + h, p_left:p_left + w] = tensor[:].clone()

        return padded_tensor

    def _calculate_layer_error(self, next_layer_error, next_layer):
        if next_layer.type == "convolutional":
            k_next = next_layer.kernel_size
            layer_error_with_padding = self._get_padded_tensor(next_layer_error, [k_next-1]*4)
            flipped_w_next = torch.flip(next_layer.w, (2, 3))

            layer_error = self._deconvolution(
                layer_error_with_padding,
                flipped_w_next,
                next_layer.output_c,
                k_next
            )

            layer_error *= self.activation.derivative(self.z)
        else:
            layer_error = next_layer_error

        return layer_error

    # def _fast_deconvolution(
    #     self,
    #     layer_error_next,
    #     filters,
    #     next_output_c,
    #     output_h,
    #     output_w,
    #     next_input_c,
    #     kernel_size
    # ):
    #     duplicated_filters = torch.zeros(
    #         next_input_c,
    #         output_h,
    #         output_w,
    #         next_output_c,
    #         kernel_size,
    #         kernel_size
    #     )
    #     for f in range(next_input_c):
    #         expanded_filter = filters[:, f].unsqueeze(0).unsqueeze(0)
    #         duplicated_filters[f] = expanded_filter.expand(output_h, output_w, -1, -1, -1)
    #
    #     separate_regions = torch.zeros(
    #         output_h,
    #         output_w,
    #         next_output_c,
    #         kernel_size,
    #         kernel_size
    #     )
    #     for i in range(output_h):
    #         for j in range(output_w):
    #             separate_regions[i][j] = layer_error_next[:, i:i + kernel_size, j:j + kernel_size]
    #
    #     expanded_regions = separate_regions.unsqueeze(0)
    #     duplicated_separate_regions = expanded_regions.expand(next_input_c, -1, -1, -1, -1, -1)
    #
    #     return torch.sum(duplicated_separate_regions * duplicated_filters, dim=(3, 4, 5))

    def _fast_deconvolution(
        self,
        layer_error_next,
        filters,
        next_output_c,
        next_kernel_size
    ):
        unfolded_regions = layer_error_next.clone().unfold(1, next_kernel_size, 1).unfold(2, next_kernel_size, 1)
        unfolded_regions = (unfolded_regions.contiguous()
                            .view(next_output_c, self.output_h * self.output_w, -1))

        reshaped_filters = filters.view(next_output_c, self.output_c, next_kernel_size * next_kernel_size)

        result = torch.einsum('abc,adc->db', unfolded_regions, reshaped_filters)

        return result.view(self.output_c, self.output_h, self.output_w)

    def _ordinary_deconvolution(
        self,
        layer_error_next,
        filters,
        next_kernel_size
    ):
        layer_error = torch.zeros((self.output_c, self.output_h, self.output_w), device=self.device)
        for f in range(self.output_c):
            for i in range(self.output_h):
                for j in range(self.output_w):
                    layer_error[f][i][j] = (
                        torch.sum(
                            layer_error_next[:, i:i + next_kernel_size, j:j + next_kernel_size]
                            * filters[:, f]
                        )
                    )
        return layer_error

    def _deconvolution(
        self,
        layer_error_next,
        filters,
        next_output_c,
        next_kernel_size
    ):
        if self.compute_mode == "fast":
            return self._fast_deconvolution(
                layer_error_next,
                filters,
                next_output_c,
                next_kernel_size
            )
        elif self.compute_mode == "ordinary":
            return self._ordinary_deconvolution(
                layer_error_next,
                filters,
                next_kernel_size
            )
        else:
            raise Exception(f"Invalid compute mode: {self.compute_mode}")

    # def _fast_convolution(
    #     self,
    #     input_image,
    #     filters,
    #     biases,
    #     input_c,
    #     output_c,
    #     output_h,
    #     output_w,
    #     kernel_size
    # ):
    #     duplicated_filters = torch.zeros(
    #         output_c,
    #         output_h,
    #         output_w,
    #         input_c,
    #         kernel_size,
    #         kernel_size
    #     )
    #     for f in range(output_c):
    #         expanded_filter = filters[f].unsqueeze(0).unsqueeze(0)
    #         duplicated_filters[f] = expanded_filter.expand(output_h, output_w, -1, -1, -1)
    #
    #     # print('duplicated_filters[0]')
    #     # print(duplicated_filters.shape)
    #     # print(duplicated_filters[0])
    #
    #     separate_regions = torch.zeros(
    #         output_h,
    #         output_w,
    #         input_c,
    #         kernel_size,
    #         kernel_size
    #     )
    #     for i in range(output_h):
    #         for j in range(output_w):
    #             separate_regions[i][j] = input_image[:, i:i + kernel_size, j:j + kernel_size]
    #
    #     expanded_regions = separate_regions.unsqueeze(0)
    #     duplicated_separate_regions = expanded_regions.expand(output_c, -1, -1, -1, -1, -1)
    #
    #     # print('duplicated_separate_regions[0]')
    #     # print(duplicated_separate_regions.shape)
    #     # print(duplicated_separate_regions[0])
    #     #
    #     # print('biases')
    #     # print(biases)
    #     # print('biases.view(-1, 1, 1)')
    #     # print(biases.view(-1, 1, 1).shape)
    #     # print(biases.view(-1, 1, 1))
    #     # print('torch.sum(duplicated_separate_regions * duplicated_filters, dim=(3, 4, 5))')
    #     # print(torch.sum(duplicated_separate_regions * duplicated_filters, dim=(3, 4, 5)).shape)
    #     # print(torch.sum(duplicated_separate_regions * duplicated_filters, dim=(3, 4, 5))[0])
    #
    #     return torch.sum(duplicated_separate_regions * duplicated_filters, dim=(3, 4, 5)) + biases.view(-1, 1, 1)

    def _fast_convolution(
        self,
        input_image
    ):
        unfolded_regions = input_image.clone().unfold(1, self.kernel_size, 1).unfold(2, self.kernel_size, 1)
        unfolded_regions = unfolded_regions.contiguous().view(self.input_c, self.output_h * self.output_w, -1)

        reshaped_filters = self.w.view(self.output_c, self.input_c, -1)

        result = torch.einsum('abc,dac->db', unfolded_regions, reshaped_filters)
        result = result.view(self.output_c, self.output_h, self.output_w)

        result += self.b.view(-1, 1, 1)

        return result

    def _ordinary_convolution(
        self,
        input_image
    ):
        z = torch.zeros_like(self.z)
        for f in range(self.output_c):
            for i in range(self.output_h):
                for j in range(self.output_w):
                    z[f][i][j] = (
                        torch.sum(input_image[:, i:i + self.kernel_size, j:j + self.kernel_size] * self.w[f])
                        + self.b[f]
                    )
        return z

    def _convolution(
        self,
        input_image
    ):
        if self.compute_mode == "fast":
            return self._fast_convolution(
                input_image
            )
        elif self.compute_mode == "ordinary":
            return self._ordinary_convolution(
                input_image
            )
        else:
            raise Exception(f"Invalid compute mode: {self.compute_mode}")

    def _fast_update_gradients(
        self,
        layer_error,
        prev_layer
    ):
        unfolded_prev_layer_a = prev_layer.a.clone().unfold(1, self.output_h, 1).unfold(2, self.output_w, 1)
        unfolded_prev_layer_a = unfolded_prev_layer_a.contiguous().view(self.input_c, self.kernel_size * self.kernel_size, -1)

        reshaped_layer_error = layer_error.view(self.output_c, -1)

        grad_w_update = torch.einsum('ab,cdb->acd', reshaped_layer_error, unfolded_prev_layer_a)
        grad_b_update = torch.einsum('f...->f', reshaped_layer_error)

        self.grad_w += grad_w_update.view(self.output_c, self.input_c, self.kernel_size, self.kernel_size)
        self.grad_b += grad_b_update.view(self.output_c)

        return

    def _ordinary_update_gradients(
        self,
        layer_error,
        prev_layer
    ):
        for f in range(self.output_c):
            layer_error_f = layer_error[f]
            for c in range(self.input_c):
                prev_layer_a_c = prev_layer.a[c]
                for m in range(self.kernel_size):
                    for n in range(self.kernel_size):
                        self.grad_w[f][c][m][n] += torch.sum(
                            layer_error_f
                            * prev_layer_a_c[m:m + self.output_h, n:n + self.output_w]
                        )
            self.grad_b[f] += torch.sum(layer_error_f)
        return

    def _update_gradients(
        self,
        layer_error,
        prev_layer
    ):
        if self.compute_mode == "fast":
            return self._fast_update_gradients(
                layer_error,
                prev_layer
            )
        elif self.compute_mode == "ordinary":
            return self._ordinary_update_gradients(
                layer_error,
                prev_layer
            )
        else:
            raise Exception(f"Invalid compute mode: {self.compute_mode}")

    def initialize(self, previous_layer, device):
        self.device = device
        if previous_layer.type == 'convolutional':
            self.input_c = previous_layer.output_c
            self.input_h = previous_layer.output_h
            self.input_w = previous_layer.output_w
        else:
            self.input_c = previous_layer.size[0]
            self.input_h = previous_layer.size[1]
            self.input_w = previous_layer.size[2]
        self.output_c = self.filters_num
        self.output_h = (self.input_h - self.kernel_size) + 1
        self.output_w = (self.input_w - self.kernel_size) + 1
        self.z = torch.zeros((self.output_c, self.output_h, self.output_w), device=self.device)
        self.a = torch.zeros((self.output_c, self.output_h, self.output_w), device=self.device)
        self.w_shape = (self.output_c, self.input_c, self.kernel_size, self.kernel_size)
        self.w = self._init_weights(self.w_shape, self.device)
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
        if previous_layer.type != 'convolutional':
            raise Exception("Flatten layer should be used only after convolutional")

        self.size = (previous_layer.output_c
                     * previous_layer.output_h
                     * previous_layer.output_w)
        self.a = torch.zeros((self.size, 1), device=device)

        return self

    def forward(self, input_data):
        self.a = input_data.flatten()
        self.a = self.a.reshape(self.a.size(0), 1)
        return

    def backward(self, layer_error, prev_layer, next_layer):
        reshape_sizes = (
            prev_layer.output_c,
            prev_layer.output_h,
            prev_layer.output_w
        )
        layer_error = (torch.matmul(next_layer.w.t(), layer_error)
                       * torch.ones_like(self.a)).reshape(reshape_sizes)

        return layer_error