# ComfyUI Qwen Image 2.1 Tools

Three custom ComfyUI nodes for Qwen Image 2.1 dynamic scheduling, fixed Turbo scheduling, and latent creation.

## Why these exist

Qwen Image 2.1's published sampling setup adjusts its sigma schedule to the target image resolution. These nodes make that behavior directly usable in ComfyUI workflows.

In the ComfyUI implementation checked on September 28, 2026, the default Qwen Image 2.1 sampling path uses a fixed shift of `0.69`, approximately the value for 1024 × 1024. The resolution-dependent reference values appear in a code comment, but the shift is not automatically recalculated from the selected image size.

The dynamic scheduler calculates that shift from the connected target latent and, with its defaults, follows the published schedule's dynamic shifting and terminal stretching. The aim is to bring the published scheduling behavior into a convenient node. The difference from a fixed shift becomes more pronounced as the image area moves away from 1024 × 1024.

The included empty latent node provides convenient width, height, and batch controls. It is optional: the scheduler also accepts a compatible latent from ComfyUI's Qwen Image 2.1 conditioning node.

ComfyUI's default workflow remains usable. This scheduler does not guarantee better images or identical results to the official pipeline: sampler choice, step count, CFG, LoRAs, and other settings still affect the result.

<details>
<summary>Code reference: Qwen's dynamic schedule and ComfyUI's fixed default</summary>

### Official Qwen Image 2.1 scheduling

The [Diffusers Qwen Image 2.1 pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py) starts with evenly spaced sigmas and calculates the shift from the target latent's spatial token count. This excerpt shows the default-sigma path; the branch for user-supplied sigmas is omitted:

```python
sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
mu = calculate_shift(
    latents.shape[1],
    self.scheduler.config.get("base_image_seq_len", 256),
    self.scheduler.config.get("max_image_seq_len", 4096),
    self.scheduler.config.get("base_shift", 0.5),
    self.scheduler.config.get("max_shift", 1.15),
)
```

The fallback values above are overridden by [Qwen Image 2.1's published scheduler configuration](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json). Relevant settings:

```json
{
  "base_image_seq_len": 256,
  "base_shift": 0.5,
  "max_image_seq_len": 8192,
  "max_shift": 0.9,
  "shift_terminal": 0.02,
  "time_shift_type": "exponential",
  "use_dynamic_shifting": true
}
```

The shift is linearly interpolated or extrapolated from those anchors. At 1024 × 1024, the target has 4096 spatial tokens and the calculated shift is approximately `0.69355`.

The pipeline passes that shift and the base sigmas to [FlowMatchEulerDiscreteScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py), whose exponential shift uses:

```python
return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
```

Here, `t` is the base sigma sequence and the exponent argument `sigma` is `1.0`. The scheduler then stretches the shifted sequence so its final nonzero value is `0.02` and appends a zero to finish sampling. Our node follows this sequence with its default settings; for a single step, it deliberately skips stretching and returns `[1, 0]`.

### ComfyUI's comment and default

In `comfy/supported_models.py`, the `QwenImage21` class contains this comment and active setting, checked on September 28, 2026:

```python
# scheduler mu at 1024x1024 (base 0.5 @ 256 tokens, max 0.9 @ 8192)
sampling_settings = {
    "multiplier": 1.0,
    "shift": 0.69,
}
```

This applies a fixed shift close to the 1024 × 1024 value. It does not calculate a new shift from the target latent's dimensions. See [ComfyUI's model defaults](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/supported_models.py).

</details>

## Included nodes

- **Qwen Image 2.1 Dynamic Scheduler** — generates a resolution-aware sigma schedule for use with **SamplerCustom**, **SamplerCustomAdvanced** or **ClownsharKSampler** with adjustable shift settings.
- **Qwen Image 2.1 Turbo Scheduler** — returns the fixed eight-step schedule for Qwen Image 2.1 Turbo, with no inputs or resolution-dependent adjustments. Found under **Qwen Image 2.1/Sampling**.
- **Qwen Image 2.1 Empty Latent** — prepares a starting latent in the format Qwen Image 2.1 needs, at your chosen size and batch count. Dimensions are rounded down to multiples of 32; the node reports the resulting dimensions.

## Installation

Place this folder in `ComfyUI/custom_nodes/ComfyUI-QwenImage21-Tools` and restart ComfyUI. Find the nodes under **Qwen Image 2.1**.

Requires a ComfyUI installation with Qwen Image 2.1 support and the appropriate models. This package adds no extra Python dependencies or model downloads.

## Basic use (standard Qwen Image 2.1)

1. Set the width, height, and batch size in **Qwen Image 2.1 Empty Latent**.
2. Connect its `latent` output to both the scheduler's `latent` input and your sampler's `latent_image` input.
3. Connect the scheduler's `sigmas` output to **SamplerCustom**, **SamplerCustomAdvanced** or **ClownsharKSampler** alongside your usual model, conditioning, and sampler connections.

Sampler choice, step count, CFG, LoRA strength, and scheduler settings are left open for experimentation. The defaults are a starting point, not a claim of the best settings for every workflow.

For reference-based image editing, use the latent from **Text Encode Qwen Image 2.1** for both the scheduler and sampler. This is an empty starting latent sized to the first resized reference image, not an encoding of the source image.

The dynamic scheduler also accepts generic **Empty Latent Image** outputs with `downscale_ratio_spacial` metadata. It calculates the Qwen grid using the same spatial scaling and rounding as ComfyUI's sampler. For example, a 1024 × 1024 generic empty latent has a 128 × 128 grid at scale 8; the scheduler uses the equivalent 64 × 64 Qwen grid (4096 tokens). The input tensor is not modified. Custom samplers must support ComfyUI's empty-latent conversion.

### Masks and inpainting

**Set Latent Noise Mask** works with either supported empty latent or a native Qwen latent. Connect its output to both the scheduler and sampler; adding the mask does not change the latent's channels or spatial metadata.

For masked inpainting with the source image as the starting content, encode the image using **VAE Encode** with the Qwen Image 2.1 VAE, then apply **Set Latent Noise Mask**. An empty latent, including the Qwen conditioning node's latent output, does not contain the source image to preserve outside the mask. Nonempty latents from other model formats are not converted.

The scheduler accepts odd spatial grid dimensions supported by ComfyUI's Qwen model. The included empty latent node still rounds requested pixel dimensions down to multiples of 32.

Without a connected latent, the scheduler assumes a 1024 × 1024 image.

## Qwen Image 2.1 Turbo

Use the separate **Qwen Image 2.1 Turbo Scheduler** with **Qwen Image 2.1 Turbo**. Unlike the standard dynamic scheduler, it has no configurable inputs and applies no shifting or terminal stretching. Each execution returns a fresh one-dimensional CPU `torch.float32` tensor:

```text
[1.0, 0.978453, 0.954180, 0.926626, 0.895080, 0.845148, 0.704534, 0.414568, 0.0]
```

The first eight values match the official model's [configured sampling sigmas](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/model_index.json). The final zero completes the eighth denoising transition, giving nine sigma values for eight steps. The [Turbo scheduler configuration](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/scheduler/scheduler_config.json) disables dynamic shifting and terminal stretching and uses an identity static shift of `1.0`. Diffusers appends the same terminal zero.

Recommended baseline:

| Setting | Value |
| --- | --- |
| Model | Qwen Image 2.1 Turbo |
| Scheduler | Qwen Image 2.1 Turbo Scheduler |
| Sampler | Euler |
| Steps | 8, determined by the fixed sigma sequence |
| CFG | 1.0 |
| Dynamic shifting | Disabled |
| Terminal stretching | Disabled |

Example connections to **SamplerCustomAdvanced**:

1. Connect **Qwen Image 2.1 Turbo Scheduler** → `sigmas` to **SamplerCustomAdvanced** → `sigmas`.
2. Select `euler` in **KSamplerSelect** and connect its `sampler` output to the sampler's `sampler` input.
3. Connect a **CFGGuider** configured with the Turbo model, conditioning, and CFG `1.0` to `guider`.
4. Connect your noise source to `noise` and a compatible Qwen Image 2.1 latent to `latent_image`.

The Turbo node produces only sigmas. Sampler and guidance configuration belong to the surrounding workflow; the node does not configure Euler or CFG automatically. Its `SIGMAS` output also works with **SamplerCustom** and other compatible custom sampling nodes. Do not pass it through another scheduler or append another zero.

## Status

Experimental. Results may vary across resolutions, samplers, step counts, LoRAs, and model variants. Settings and behavior may change as testing continues. Feedback and comparisons are welcome.

## License

This node package is licensed under the [MIT License](LICENSE). Referenced third-party code and model weights retain their respective licenses.
