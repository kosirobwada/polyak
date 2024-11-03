# Experiment Results

This directory contains the results of the experiments from our paper.

Each figure represents critical findings and visualizations discussed in the paper. Refer to the paper for a detailed analysis and interpretation of these results.

- [Experiment Results](#experiment-results)
  - [ResNet-18 on CIFAR100](#resnet-18-on-cifar100)
  - [ResNet-18 on CIFAR10 and CIFAR100 using DOUBLING, TRIPLING, and QUADRUPLING BATCH SIZES](#resnet-18-on-cifar10-and-cifar100-using-doubling-tripling-and-quadrupling-batch-sizes)
  - [Wide-ResNet-28-10 on CIFAR100](#wide-resnet-28-10-on-cifar100)
  - [ResNet-18 on Tiny ImageNet](#resnet-18-on-tiny-imagenet)

---

## ResNet-18 on CIFAR100

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_1/scheduler_1.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_1/norm_1.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_1/loss_1.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_1/test_1.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 1:** (a) Decaying learning rates (constant, diminishing, cosine, linear, and polynomial) and constant batch size, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train ResNet-18 on CIFAR100 dataset.

---

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_2/scheduler_2.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_2/norm_2.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_2/loss_2.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_2/test_2.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 2:** (a) Decaying learning rates and doubly increasing batch size every 30 epochs, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train ResNet-18 on CIFAR100 dataset.

---

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_3/scheduler_3.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_3/norm_3.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_3/loss_3.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_3/test_3.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 3:** (a) Increasing learning rates ($η_{max}$ = 0.2, 0.5, 1.0) and doubly increasing batch size every 30 epochs, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train ResNet-18 on CIFAR100 dataset.

---

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_4/scheduler_4.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_4/norm_4.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_4/loss_4.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_4/test_4.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 4:** (a) Warm-up learning rates and doubly increasing batch size every 30 epochs, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train ResNet-18 on CIFAR100 dataset.

---

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_5/scheduler_5.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_5/norm_5.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_5/loss_5.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_5/test_5.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 5:** (a) Increasing learning rates and increasing batch sizes based on $\delta$ = 2, 3, 4, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train ResNet-18 on CIFAR100 dataset.

---

## ResNet-18 on CIFAR10 and CIFAR100 using DOUBLING, TRIPLING, and QUADRUPLING BATCH SIZES

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_6/scheduler_d_1.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_6/norm_d_1.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_6/loss_d_1.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_6/test_d_1.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 6:** (a) Increasing learning rates and doubling, tripling, and quadrupling batch sizes (($\delta$, $\gamma$) = (2, 1.4), (3, 1.7), (4, 1.9) satisfying $\sqrt{\delta}$ > $\gamma$) every 100 epochs, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train ResNet-18 on CIFAR10 dataset.

---

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_7/scheduler_d_2.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_7/norm_d_2.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_7/loss_d_2.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_7/test_d_2.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 7:** (a) Increasing learning rates and doubling, tripling, and quadrupling batch sizes (($\delta$, $\gamma$) = (2, 1.4), (3, 1.7), (4, 1.9) satisfying $\sqrt{\delta}$ > $\gamma$) every 100 epochs, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train ResNet-18 on CIFAR100 dataset.

---

## Wide-ResNet-28-10 on CIFAR100

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_8/scheduler_1_1.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_8/norm_1_1.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_8/loss_1_1.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_8/test_1_1.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 8:** (a) Decaying learning rates (constant, diminishing, cosine, linear, and polynomial) and constant batch size, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train Wide-ResNet-28-10 on CIFAR100 dataset.

---

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_9/scheduler_2_1.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_9/norm_2_1.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_9/loss_2_1.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_9/test_2_1.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 9:** (a) Decaying learning rates and increasing batch size every 30 epochs, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train Wide-ResNet-28-10 on CIFAR100 dataset.

---

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_10/scheduler_3_1.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_10/norm_3_1.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_10/loss_3_1.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_10/test_3_1.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 10:** (a) Increasing learning rates ($\eta_{max}$ = 0.2, 0.5, 1.0) and increasing batch size every 30 epochs, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train Wide-ResNet-28-10 on CIFAR100 dataset.

---

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_11/scheduler_4_1.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_11/norm_4_1.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_11/loss_4_1.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_11/test_4_1.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 11:** (a) Warm-up learning rates and increasing batch size every 30 epochs, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train Wide-ResNet-28-10 on CIFAR100 dataset.

---

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_12/scheduler_5_1.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_12/norm_5_1.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_12/loss_5_1.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_12/test_5_1.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 12:** (a) Increasing learning rates and increasing batch sizes based on $\delta$ = 2, 3, 4, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train Wide-ResNet-28-10 on CIFAR100 dataset.

---

## ResNet-18 on Tiny ImageNet

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_13/scheduler_1_2.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_13/norm_1_2.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_13/loss_1_2.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_13/test_1_2.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 13:** (a) Decaying learning rates (constant, diminishing, cosine, linear, and polynomial) and constant batch size, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train ResNet-18 on Tiny ImageNet dataset.

---

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_14/scheduler_2_2.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_14/norm_2_2.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_14/loss_2_2.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_14/test_2_2.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 14:** (a) Decaying learning rates and increasing batch size every 30 epochs, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train ResNet-18 on Tiny ImageNet dataset.

---

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_15/scheduler_3_2.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_15/norm_3_2.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_15/loss_3_2.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_15/test_3_2.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 15:** (a) Increasing learning rates ($\eta_{max}$ = 0.2, 0.5, 1.0) and increasing batch size every 30 epochs, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train ResNet-18 on Tiny ImageNet dataset.

---

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_16/scheduler_4_2.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_16/norm_4_2.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_16/loss_4_2.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_16/test_4_2.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 16:** (a) Warm-up learning rates and increasing batch size every 30 epochs, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train ResNet-18 on Tiny ImageNet dataset.

---

<table align="center">
    <tr>
        <td align="center">
            <img src="graph/figure_17/scheduler_5_2.png" width="100%" /><br>
            (a) Learning rate η_t and batch size b versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_17/norm_5_2.png" width="100%" /><br>
            (b) Full gradient norm ||∇f(θ_e)|| versus epochs
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="graph/figure_17/loss_5_2.png" width="100%" /><br>
            (c) Empirical loss f(θ_e) versus epochs
        </td>
        <td align="center">
            <img src="graph/figure_17/test_5_2.png" width="100%" /><br>
            (d) Test accuracy score versus epochs
        </td>
    </tr>
</table>

**Figure 17:** (a) Increasing learning rates and increasing batch sizes based on $\delta$ = 2, 3, 4, (b) full gradient norm of empirical loss, (c) empirical loss value, and (d) accuracy score in testing for SGD to train ResNet-18 on Tiny ImageNet dataset.

---
