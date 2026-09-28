import math

import torch
import comfy.model_management


# -----------------------------------------------------------------------------
# Qwen Image 2.1 constants
# -----------------------------------------------------------------------------

QWEN21_LATENT_CHANNELS = 64
QWEN21_VAE_SCALE_FACTOR = 16

# Qwen Image 2.1's official Diffusers pipeline expects pixel dimensions
# divisible by VAE scale * 2 = 32.
QWEN21_PIXEL_MULTIPLE = 32

# Default target token count when no latent is supplied:
# 1024 / 16 = 64
# 64 * 64 = 4096
DEFAULT_SEQ_LEN = 4096


# -----------------------------------------------------------------------------
# Qwen Image 2.1 Empty Latent
# -----------------------------------------------------------------------------

class QwenImage21EmptyLatent:
    """
    Create a native empty latent for Qwen Image 2.1.

    Qwen Image 2.1 uses:
      - 64 latent channels
      - 16x spatial VAE downscale
      - pixel dimensions aligned to multiples of 32

    Width and height are rounded DOWN to the nearest multiple of 32.
    The returned actual_width / actual_height outputs show the dimensions
    that will really be represented by the latent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": QWEN21_PIXEL_MULTIPLE,
                        "max": 16384,
                        "step": QWEN21_PIXEL_MULTIPLE,
                        "tooltip": (
                            "Requested output width in pixels. "
                            "Qwen Image 2.1 uses dimensions divisible by 32. "
                            "If a linked value is not divisible by 32, it is "
                            "rounded down to the nearest multiple of 32."
                        ),
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": QWEN21_PIXEL_MULTIPLE,
                        "max": 16384,
                        "step": QWEN21_PIXEL_MULTIPLE,
                        "tooltip": (
                            "Requested output height in pixels. "
                            "Qwen Image 2.1 uses dimensions divisible by 32. "
                            "If a linked value is not divisible by 32, it is "
                            "rounded down to the nearest multiple of 32."
                        ),
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "Number of Qwen Image 2.1 empty latents to create.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "latent",
        "latent_width",
        "latent_height",
        "actual_width",
        "actual_height",
    )

    FUNCTION = "generate"
    CATEGORY = "Qwen Image 2.1/Latent"

    DESCRIPTION = (
        "Creates a native Qwen Image 2.1 empty latent using 64 channels, "
        "16x VAE downscaling and 32-pixel output alignment."
    )

    @staticmethod
    def _round_down_to_multiple(value, multiple):
        value = int(value)

        if value < multiple:
            raise ValueError(
                f"Qwen Image 2.1: dimension must be at least {multiple}, "
                f"but received {value}."
            )

        return (value // multiple) * multiple

    def generate(self, width, height, batch_size=1):
        width = int(width)
        height = int(height)
        batch_size = int(batch_size)

        if batch_size < 1:
            raise ValueError(
                f"Qwen Image 2.1: batch_size must be at least 1, "
                f"but received {batch_size}."
            )

        actual_width = self._round_down_to_multiple(
            width,
            QWEN21_PIXEL_MULTIPLE,
        )

        actual_height = self._round_down_to_multiple(
            height,
            QWEN21_PIXEL_MULTIPLE,
        )

        latent_width = actual_width // QWEN21_VAE_SCALE_FACTOR
        latent_height = actual_height // QWEN21_VAE_SCALE_FACTOR

        # Because output dimensions are multiples of 32 and the VAE scale
        # factor is 16, both latent dimensions must be even.
        if latent_width % 2 != 0 or latent_height % 2 != 0:
            raise RuntimeError(
                "Internal Qwen Image 2.1 latent alignment error: "
                f"latent size became {latent_width}x{latent_height}."
            )

        latent = torch.zeros(
            (
                batch_size,
                QWEN21_LATENT_CHANNELS,
                latent_height,
                latent_width,
            ),
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )

        return (
            {
                "samples": latent,
                "downscale_ratio_spacial": QWEN21_VAE_SCALE_FACTOR,
            },
            latent_width,
            latent_height,
            actual_width,
            actual_height,
        )


# -----------------------------------------------------------------------------
# Qwen Image 2.1 Dynamic Scheduler
# -----------------------------------------------------------------------------

class QwenImage21DynamicScheduler:
    """
    Resolution-aware sigma scheduler for Qwen Image 2.1.

    The target image sequence length is calculated from the supplied native
    Qwen 2.1 latent:

        seq_len = latent_height * latent_width

    The default shift parameters correspond to Qwen Image 2.1's released
    scheduler configuration.

    The base inference sigma sequence follows the Qwen Image 2.1 Diffusers
    pipeline convention:

        linspace(1.0, 1.0 / steps, steps)

    Dynamic shifting is then applied and, when enabled, the nonzero schedule
    is stretched so its last value equals shift_terminal.

    Finally, exactly one zero sigma is appended for sampler termination.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": (
                    "INT",
                    {
                        "default": 40,
                        "min": 1,
                        "max": 10000,
                        "tooltip": (
                            "Number of denoising steps. Qwen Image 2.1 "
                            "defaults to 40. With steps=1, terminal stretching "
                            "is intentionally skipped because a single sigma "
                            "cannot remain 1.0 and simultaneously end at the "
                            "terminal value."
                        ),
                    },
                ),

                "base_shift": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.0001,
                        "tooltip": (
                            "Dynamic shift value at base_image_seq_len. "
                            "Qwen Image 2.1 default: 0.5."
                        ),
                    },
                ),

                "max_shift": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.0001,
                        "tooltip": (
                            "Dynamic shift value at max_image_seq_len. "
                            "Qwen Image 2.1 default: 0.9."
                        ),
                    },
                ),

                "base_image_seq_len": (
                    "INT",
                    {
                        "default": 256,
                        "min": 1,
                        "max": 1000000,
                        "tooltip": (
                            "Lower image-token anchor used to calculate mu. "
                            "Qwen Image 2.1 default: 256."
                        ),
                    },
                ),

                "max_image_seq_len": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 2,
                        "max": 1000000,
                        "tooltip": (
                            "Upper image-token anchor used to calculate mu. "
                            "Must be larger than base_image_seq_len. "
                            "Qwen Image 2.1 default: 8192."
                        ),
                    },
                ),

                "shift_terminal": (
                    "FLOAT",
                    {
                        "default": 0.02,
                        "min": 0.000001,
                        "max": 0.999999,
                        "step": 0.0001,
                        "tooltip": (
                            "Target value of the final nonzero sigma when "
                            "terminal stretching is enabled. "
                            "Qwen Image 2.1 default: 0.02."
                        ),
                    },
                ),

                "time_shift_type": (
                    ["exponential", "linear"],
                    {
                        "default": "exponential",
                        "tooltip": (
                            "Dynamic timestep shift function. "
                            "Qwen Image 2.1 uses exponential. "
                            "Linear is provided only for experimentation."
                        ),
                    },
                ),

                "stretch_to_terminal": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Rescale the nonzero sigma schedule so that its "
                            "final value equals shift_terminal. "
                            "Skipped automatically when steps=1."
                        ),
                    },
                ),

                "clamp_mu": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "If enabled, clamp the calculated mu between "
                            "base_shift and max_shift. Qwen's normal dynamic "
                            "shift calculation does not clamp, so False is "
                            "the reference behavior. This option is useful "
                            "for experimentation at extreme resolutions."
                        ),
                    },
                ),
            },

            "optional": {
                "latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Native Qwen Image 2.1 target latent. "
                            "Expected shape: [batch, 64, height, width]. "
                            "Only target image spatial tokens are counted; "
                            "text and reference-image tokens are not included."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("SIGMAS", "FLOAT", "INT")
    RETURN_NAMES = ("sigmas", "mu", "seq_len")

    FUNCTION = "get_sigmas"
    CATEGORY = "Qwen Image 2.1/Sampling"

    DESCRIPTION = (
        "Creates a resolution-aware Qwen Image 2.1 sigma schedule from the "
        "target latent's spatial token count."
    )

    # -------------------------------------------------------------------------
    # Validation helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_finite(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Qwen Image 2.1 Scheduler: {name} must be a real number."
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                f"Qwen Image 2.1 Scheduler: {name} must be finite, "
                f"but received {value}."
            )

        return value

    @staticmethod
    def _get_seq_len(latent):
        """
        Extract the target image sequence length from a native Qwen 2.1 latent.

        Qwen Image 2.1 consumes the spatial latent grid directly, so:

            seq_len = H * W

        If no latent is supplied, 4096 is used, corresponding to a
        1024x1024 image:
            1024 / 16 = 64
            64 * 64 = 4096
        """

        if latent is None:
            return DEFAULT_SEQ_LEN

        if not isinstance(latent, dict):
            raise ValueError(
                "Qwen Image 2.1 Scheduler: connected LATENT must be a "
                "dictionary containing a 'samples' tensor."
            )

        if "samples" not in latent:
            raise ValueError(
                "Qwen Image 2.1 Scheduler: connected LATENT does not contain "
                "a 'samples' tensor."
            )

        samples = latent["samples"]

        if not isinstance(samples, torch.Tensor):
            raise ValueError(
                "Qwen Image 2.1 Scheduler: latent['samples'] must be a "
                "torch.Tensor."
            )

        if samples.ndim != 4:
            raise ValueError(
                "Qwen Image 2.1 Scheduler: expected a 4D image latent with "
                "shape [batch, channels, height, width], but received shape "
                f"{tuple(samples.shape)}."
            )

        batch, channels, height, width = samples.shape

        if batch < 1:
            raise ValueError(
                "Qwen Image 2.1 Scheduler: latent batch dimension is empty."
            )

        if channels != QWEN21_LATENT_CHANNELS:
            raise ValueError(
                "Qwen Image 2.1 Scheduler: expected a native Qwen Image 2.1 "
                f"latent with {QWEN21_LATENT_CHANNELS} channels, but received "
                f"{channels}. Use 'Qwen Image 2.1 Empty Latent' or the latent "
                "output from the Qwen Image 2.1 conditioning node instead of "
                "the generic Empty Latent Image."
            )

        if height < 1 or width < 1:
            raise ValueError(
                "Qwen Image 2.1 Scheduler: latent spatial dimensions must "
                f"be positive, but received {width}x{height}."
            )

        # Official Qwen Image 2.1 pixel dimensions are aligned to 32.
        # With a 16x VAE, this means the latent H/W should be even.
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                "Qwen Image 2.1 Scheduler: native Qwen 2.1 latent width and "
                "height should be even, corresponding to output dimensions "
                "divisible by 32. Received latent size "
                f"{width}x{height}, corresponding to approximately "
                f"{width * 16}x{height * 16} pixels."
            )

        seq_len = int(height) * int(width)

        if seq_len < 1:
            raise ValueError(
                "Qwen Image 2.1 Scheduler: calculated seq_len is invalid."
            )

        return seq_len

    # -------------------------------------------------------------------------
    # Dynamic shift
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_mu(
        seq_len,
        base_image_seq_len,
        max_image_seq_len,
        base_shift,
        max_shift,
        clamp_mu,
    ):
        base_image_seq_len = int(base_image_seq_len)
        max_image_seq_len = int(max_image_seq_len)

        if base_image_seq_len < 1:
            raise ValueError(
                "Qwen Image 2.1 Scheduler: base_image_seq_len must be >= 1."
            )

        if max_image_seq_len <= base_image_seq_len:
            raise ValueError(
                "Qwen Image 2.1 Scheduler: max_image_seq_len must be larger "
                "than base_image_seq_len. Received "
                f"{base_image_seq_len} and {max_image_seq_len}."
            )

        base_shift = QwenImage21DynamicScheduler._validate_finite(
            "base_shift",
            base_shift,
        )

        max_shift = QwenImage21DynamicScheduler._validate_finite(
            "max_shift",
            max_shift,
        )

        # Same linear shift interpolation/extrapolation used by the Qwen
        # Diffusers pipeline's calculate_shift().
        slope = (
            (max_shift - base_shift)
            / (max_image_seq_len - base_image_seq_len)
        )

        intercept = base_shift - slope * base_image_seq_len

        mu = float(seq_len) * slope + intercept

        if not math.isfinite(mu):
            raise ValueError(
                "Qwen Image 2.1 Scheduler: calculated mu is non-finite. "
                "Check the shift values and sequence-length anchors."
            )

        if clamp_mu:
            low = min(base_shift, max_shift)
            high = max(base_shift, max_shift)
            mu = max(low, min(high, mu))

        return float(mu)

    # -------------------------------------------------------------------------
    # Numerically stable time shifting
    # -------------------------------------------------------------------------

    @staticmethod
    def _time_shift_exponential(mu, t):
        """
        Numerically stable form of:

            exp(mu) / (exp(mu) + (1/t - 1))

        Algebraically:

            sigmoid(logit(t) + mu)

        This avoids math.exp(mu) overflow for extreme mu values.
        """

        t = t.to(dtype=torch.float64)

        if torch.any(t <= 0.0) or torch.any(t > 1.0):
            raise ValueError(
                "Qwen Image 2.1 Scheduler: base sigmas must lie in (0, 1]."
            )

        # logit(t) = log(t) - log(1 - t)
        #
        # Handle t == 1 separately because log1p(-1) = -inf.
        result = torch.empty_like(t, dtype=torch.float64)

        at_one = t >= 1.0
        below_one = ~at_one

        result[at_one] = 1.0

        if torch.any(below_one):
            tb = t[below_one]

            logit_t = torch.log(tb) - torch.log1p(-tb)

            result[below_one] = torch.sigmoid(
                logit_t + float(mu)
            )

        return result

    @staticmethod
    def _time_shift_linear(mu, t):
        """
        Stable form of the Diffusers linear time shift:

            mu / (mu + (1/t - 1))

        rewritten as:

            (mu * t) / (mu * t + 1 - t)

        Linear shifting requires mu > 0.
        """

        if mu <= 0.0:
            raise ValueError(
                "Qwen Image 2.1 Scheduler: linear time shifting requires "
                f"mu > 0, but calculated mu={mu:.8g}. "
                "Use exponential shifting, enable clamp_mu, or choose "
                "shift/anchor values that produce a positive mu."
            )

        t = t.to(dtype=torch.float64)

        if torch.any(t <= 0.0) or torch.any(t > 1.0):
            raise ValueError(
                "Qwen Image 2.1 Scheduler: base sigmas must lie in (0, 1]."
            )

        numerator = float(mu) * t
        denominator = numerator + (1.0 - t)

        if torch.any(denominator <= 0.0):
            raise ValueError(
                "Qwen Image 2.1 Scheduler: linear time shift produced an "
                "invalid denominator."
            )

        return numerator / denominator

    # -------------------------------------------------------------------------
    # Terminal stretching
    # -------------------------------------------------------------------------

    @staticmethod
    def _stretch_to_terminal(base_sigmas, terminal, log_shift):
        """
        Apply shifting and terminal stretching without subtracting shifted
        sigmas from 1, which loses precision for large positive shifts.

        With u = (1 - t) / t, a = exp(log_shift), and U = u[-1],
        the stretched sigma is:

            terminal + (1 - terminal) * a * (U - u) / (U * (a + u))

        Exponential shifting uses log_shift = mu; linear uses log(mu).

        For a one-step schedule, stretching is intentionally skipped:
        a single value cannot simultaneously remain 1.0 and become terminal.
        """

        if base_sigmas.numel() <= 1:
            return base_sigmas

        terminal = float(terminal)

        if not math.isfinite(terminal):
            raise ValueError(
                "Qwen Image 2.1 Scheduler: shift_terminal must be finite."
            )

        if not (0.0 < terminal < 1.0):
            raise ValueError(
                "Qwen Image 2.1 Scheduler: when terminal stretching is "
                "enabled, shift_terminal must be greater than 0 and less "
                f"than 1. Received {terminal}."
            )

        # Exclude t=1 so an underflowed exp(log_shift) cannot cause 0/0.
        u = (1.0 - base_sigmas[1:]) / base_sigmas[1:]
        last_u = u[-1]
        if log_shift >= 0.0:
            remaining = ((last_u - u) / last_u) / (
                1.0 + u * math.exp(-log_shift)
            )
        else:
            shift = math.exp(log_shift)
            remaining = ((last_u - u) / last_u) * (shift / (shift + u))

        stretched = torch.empty_like(base_sigmas)
        stretched[0] = 1.0
        stretched[1:] = terminal + (1.0 - terminal) * remaining

        return stretched

    # -------------------------------------------------------------------------
    # Final schedule validation
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_schedule(sigmas):
        if sigmas.ndim != 1 or sigmas.numel() < 1:
            raise ValueError(
                "Qwen Image 2.1 Scheduler: generated sigma schedule is empty "
                "or malformed."
            )

        if not torch.all(torch.isfinite(sigmas)):
            raise ValueError(
                "Qwen Image 2.1 Scheduler: generated sigma schedule contains "
                "NaN or infinite values."
            )

        tolerance = 1e-10

        if torch.any(sigmas < -tolerance):
            raise ValueError(
                "Qwen Image 2.1 Scheduler: generated sigma schedule contains "
                "negative values."
            )

        if torch.any(sigmas > 1.0 + tolerance):
            raise ValueError(
                "Qwen Image 2.1 Scheduler: generated sigma schedule contains "
                "values greater than 1."
            )

        # Denoising sigmas must be monotonically non-increasing.
        if sigmas.numel() > 1:
            if torch.any(sigmas[1:] > sigmas[:-1] + tolerance):
                raise ValueError(
                    "Qwen Image 2.1 Scheduler: generated sigmas are not "
                    "monotonically decreasing."
                )

    # -------------------------------------------------------------------------
    # Node execution
    # -------------------------------------------------------------------------

    def get_sigmas(
        self,
        steps,
        base_shift,
        max_shift,
        base_image_seq_len,
        max_image_seq_len,
        shift_terminal,
        time_shift_type,
        stretch_to_terminal,
        clamp_mu,
        latent=None,
    ):
        steps = int(steps)

        if steps < 1:
            raise ValueError(
                "Qwen Image 2.1 Scheduler: steps must be at least 1."
            )

        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError(
                "Qwen Image 2.1 Scheduler: time_shift_type must be either "
                f"'exponential' or 'linear', but received "
                f"{time_shift_type!r}."
            )

        seq_len = self._get_seq_len(latent)

        mu = self._compute_mu(
            seq_len=seq_len,
            base_image_seq_len=base_image_seq_len,
            max_image_seq_len=max_image_seq_len,
            base_shift=base_shift,
            max_shift=max_shift,
            clamp_mu=bool(clamp_mu),
        )

        # Qwen Image 2.1's Diffusers pipeline explicitly constructs its
        # inference sigmas as:
        #
        # np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        #
        # We compute in float64 for numerical robustness and return float32.
        base_sigmas = torch.linspace(
            1.0,
            1.0 / float(steps),
            steps,
            dtype=torch.float64,
            device="cpu",
        )

        if time_shift_type == "exponential":
            sigmas = self._time_shift_exponential(
                mu,
                base_sigmas,
            )
        elif time_shift_type == "linear":
            sigmas = self._time_shift_linear(
                mu,
                base_sigmas,
            )
        else:
            # Defensive fallback for direct/non-UI calls.
            raise ValueError(
                f"Unsupported time_shift_type: {time_shift_type!r}"
            )

        if stretch_to_terminal:
            shift_terminal = self._validate_finite(
                "shift_terminal",
                shift_terminal,
            )

            # steps=1 intentionally remains [1.0] before the terminal zero.
            if steps > 1:
                sigmas = self._stretch_to_terminal(
                    base_sigmas,
                    shift_terminal,
                    mu if time_shift_type == "exponential" else math.log(mu),
                )

        sigmas = sigmas.to(dtype=torch.float32)
        self._validate_schedule(sigmas)

        if torch.any(sigmas <= 0.0):
            raise ValueError(
                "Qwen Image 2.1 Scheduler: denoising sigmas underflowed to "
                "zero in float32. Use less extreme shift values or enable "
                "terminal stretching with a representable positive terminal."
            )

        # Every denoising sigma is positive; append the only termination zero.
        final_sigmas = torch.cat((sigmas, sigmas.new_zeros(1)), dim=0)

        return (
            final_sigmas,
            float(mu),
            int(seq_len),
        )


# -----------------------------------------------------------------------------
# ComfyUI registration
# -----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "QwenImage21EmptyLatent": QwenImage21EmptyLatent,
    "QwenImage21DynamicScheduler": QwenImage21DynamicScheduler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImage21EmptyLatent": "Qwen Image 2.1 Empty Latent",
    "QwenImage21DynamicScheduler": "Qwen Image 2.1 Dynamic Scheduler",
}
