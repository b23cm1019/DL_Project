# EXPERIMENTAL SETUP & METHODOLOGY

To accommodate extreme time constraints (4 days) and consumer-grade hardware limitations (single NVIDIA RTX A5000 with 25.4GB VRAM), our experimental methodology diverges from the original paper's setup while carefully maintaining a fair, isolatable ablation format. You can find out more details about how we scaled our configuration down securely below:

## 1. Experimental Roadmap (The "4 Rows")

Our project aims to show the isolated impact of our proposed **Multi-Label BCE Loss** and **Semantic Soft Prompt Injection**. To do this, we run variants on the CDD-11 Mixed Degradation Dataset:

*   **Row A:** Download + test the provided `dcpt_nafnet_cdd.pth`
    *   *Purpose: Provides the absolute Reference Ceiling for our architecture (NAFNet trained for 750k iterations full-scale).*
*   **Row B:** Pretrain (single-label, 25k iters) + finetune (100k iters) + test
    *   *Purpose: Establishes our local limited-iteration baseline. We replicate the authors' FocalLoss approach natively within our constraint bounds.*
*   **Row C:** Pretrain (multi-label BCE, 25k iters) + finetune (100k iters) + test
    *   *Purpose: Tests Contribution 1. Identical bounds to Row B, but uses compound decomposition multi-hot BCE as the supervision signal rather than single-label FocalLoss.*
*   **Row D (Row C + prompt):** Add prompt injection + finetune (100k iters) + test
    *   *Purpose: Tests Contribution 2. We take Row C's Pretrained bounds, but insert the frozen predicted classifier output actively backward into the NAFNet middle-block as a conditioning prompt.*

---

## 2. Updated Project Settings vs Paper Parameters

Given our severe hardware constraints of using just 1 GPU without gradient accumulation, we radically overhauled the Hyperparameters to stabilize training.

| Parameter | Original Paper | Our Project Setup | Rationale / Change Details |
| :--- | :--- | :--- | :--- |
| **Pretrain Iters** | 100k | **25k** | Heavy reduction to fit within the 4-day time limit. |
| **Finetune Iters** | 750k | **100k** | Heavy reduction to fit within the 4-day time limit. |
| **Batch Size** | 32 (4 GPUs × 8 each) | **8** | Restricted to `8` total due to 1 GPU, skipping gradient accumulation completely. |
| **LR (encoder)** | 3e-4 | **1e-4** | Compensates for the huge drop in effective batch-size to prevent wildly unstable gradients. |
| **LR (decoder)** | 1e-4 | **5e-5** | Scaled proportionately downwards to ensure loss doesn't diverge in classifier optimization. |
| **Patch Size** | 128x128 | **128x128** | *(remains identical)* |
| **Architecture** | NAFNet, SwinIR, Restormer, PromptIR | **NAFNet only** | NAFNet requires the least memory footprint while consistently preserving strong SSIM outputs. |

We will synthesize these results back into a formal writeup to demonstrate that our multi-label changes yield proportionate, albeit constrained, boosts over standard self-supervision.
