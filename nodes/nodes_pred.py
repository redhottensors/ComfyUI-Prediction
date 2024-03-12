import comfy
import torch
import latent_preview
import sys

class CustomNoisePredictor(torch.nn.Module):
    def __init__(self, model, pred, conds):
        super().__init__()

        self.inner_model = model
        self.pred = pred
        self.conds = conds

    def apply_model(self, x, timestep, cond, uncond, cond_scale, model_options={}, seed=None):
        self.pred.reset_cache() # cheap, just in case

        try:
            pred = self.pred.predict_noise(x, timestep, self.inner_model, self.conds, model_options, seed)
        finally:
            self.pred.reset_cache()

        return pred

    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)

class SamplerCustomPrediction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "noise_prediction": ("PREDICTION",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/prediction"

    def sample(self, model, add_noise, noise_seed, sampler, sigmas, latent_image, noise_prediction):
        latent_samples = latent_image["samples"]

        if not add_noise:
            torch.manual_seed(noise_seed)

            noise = torch.zeros(
                latent_samples.size(),
                dtype=latent_samples.dtype,
                layout=latent_samples.layout,
                device="cpu"
            )
        else:
            batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
            noise = comfy.sample.prepare_noise(latent_samples, noise_seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent_image:
            noise_mask = latent_image["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

        samples = sample_pred(
            model, noise, noise_prediction, sampler, sigmas, latent_samples,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=noise_seed
        )

        out = latent_image.copy()
        out["samples"] = samples

        if "x0" in x0_output:
            out_denoised = latent_image.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out

        return (out, out_denoised)

def sample_pred(model, noise, predictor, sampler, sigmas, latent, noise_mask=None, callback=None, disable_pbar=False, seed=None):
    if noise_mask is not None:
        noise_mask = comfy.sample.prepare_mask(noise_mask, noise.shape, model.load_device)

    dtype = model.model_dtype()
    device = model.load_device

    models = predictor.get_models()
    conds = predictor.get_conds()
    preds = predictor.get_preds()

    n_samples = 0
    for pred in preds:
        n_samples += pred.n_samples()

    inference_memory = model.memory_required([noise.shape[0] * n_samples] + list(noise.shape[1:]))

    for addtl in models:
        if "inference_memory_requirements" in addtl:
            inference_memory += addtl.inference_memory_requirements(dtype)

    comfy.model_management.load_models_gpu(models | set([model]), inference_memory)

    noise = noise.to(device)
    latent = latent.to(device)
    sigmas = sigmas.to(device)

    for name, cond in conds.items():
        conds[name] = cond[:]

    for cond in conds.values():
        comfy.samplers.resolve_areas_and_cond_masks(cond, noise.shape[2], noise.shape[3], device)

    for cond in conds.values():
        comfy.samplers.calculate_start_end_timesteps(model.model, cond)

    if latent is not None:
        latent = model.model.process_latent_in(latent)

    if hasattr(model.model, "extra_conds"):
        for name, cond in conds.items():
            conds[name] = comfy.samplers.encode_model_conds(
                model.model.extra_conds,
                cond, noise, device, name,
                latent_image=latent,
                denoise_mask=noise_mask,
                seed=seed
            )

    # ensure each cond area corresponds with all other areas
    for name1, cond1 in conds.items():
        for name2, cond2 in conds.items():
            if name2 == name1:
                continue

            for c1 in cond1:
                comfy.samplers.create_cond_with_same_area_if_none(cond2, c1)

    # TODO: support controlnet how?

    predictor_model = CustomNoisePredictor(model.model, predictor, conds)
    extra_args = {
        "cond": None, "uncond": None, "cond_scale": None,
        "model_options": model.model_options, "seed": seed
    }

    samples = sampler.sample(predictor_model, sigmas, extra_args, callback, noise, latent, noise_mask, disable_pbar)
    samples = model.model.process_latent_out(samples.to(torch.float32))
    samples = samples.to(comfy.model_management.intermediate_device())

    comfy.sample.cleanup_additional_models(models)
    return samples

class NoisePredictor:
    OUTPUTS = { "prediction": "PREDICTION" }

    def get_models(self):
        """Returns all additional models transitively used by this predictor."""
        return set()

    def get_conds(self):
        """Returns all conditionings transitively defined by this predictor."""
        return {}

    def get_preds(self):
        """Returns all NoisePredcitors transitively used by this predictor, including itself."""
        return {self}

    def n_samples(self):
        """Returns the number of times a model will be sampled directly by this predictor."""
        return 0

    def predict_noise(self, x, timestep, model, conds, model_options, seed):
        raise Exception("not implemented")

    def reset_cache(self):
        """Transitively resets all cached predictions."""
        pass

    def get_models_from_conds(self):
        models = set()

        for cond in self.get_conds():
            for cnet in comfy.sample.get_models_from_cond(cond, "control"):
                models |= cnet.get_models()

            for gligen in comfy.sample.get_models_from_cond(cond, "gligen"):
                models |= [x[1] for x in gligen]

        return models

    @staticmethod
    def merge_conds(*args):
        merged = {}

        for arg in args:
            for name, cond in arg.items():
                if name not in merged:
                    merged[name] = cond
                elif merged[name] != cond:
                    raise Exception(f"Conditioning \"{name}\" is not unique.")

        return merged

class CachingNoisePredictor(NoisePredictor):
    def __init__(self):
        self.cached_prediction = None

    def predict_noise(self, x, timestep, model, conds, model_options, seed):
        if self.cached_prediction is None:
            self.cached_prediction = self.predict_noise_uncached(x, timestep, model, conds, model_options, seed)

        return self.cached_prediction

    def predict_noise_uncached(self, x, timestep, model, conds, model_options, seed):
        raise Exception("not implemented")

    def reset_cache(self):
        self.cached_prediction = None

class ConditionedPredictor(CachingNoisePredictor):
    INPUTS = {
        "required": {
            "conditioning": ("CONDITIONING",),
            "name": ("STRING", {
                "multiline": False,
                "default": "positive"
            }),
        }
    }

    def __init__(self, conditioning, name):
        self.cond = comfy.sample.convert_cond(conditioning)
        self.cond_name = name

    def get_conds(self):
        return { self.cond_name: self.cond }

    def get_models(self):
        return self.get_models_from_conds()

    def n_samples(self):
        return 1

    def predict_noise_uncached(self, x, timestep, model, conds, model_options, seed):
        pred, _ = comfy.samplers.calc_cond_uncond_batch(
            model,
            conds[self.cond_name],
            None,
            x,
            timestep,
            model_options
        )

        return pred

class CombinePredictor(NoisePredictor):
    INPUTS = {
        "required": {
            "prediction_A": ("PREDICTION",),
            "prediction_B": ("PREDICTION",),
            "operation": ([
                "A + B",
                "A - B",
                "A * B",
                "A / B",
                "A proj B",
                "A oproj B",
                "min(A, B)",
                "max(A, B)",
            ],)
        }
    }

    def __init__(self, prediction_A, prediction_B, operation):
        self.lhs = prediction_A
        self.rhs = prediction_B

        match operation:
            case "A + B":
                self.op = torch.add
            case "A - B":
                self.op = torch.sub
            case "A * B":
                self.op = torch.mul
            case "A / B":
                self.op = torch.div
            case "A proj B":
                self.op = proj
            case "A oproj B":
                self.op = oproj
            case "min(A, B)":
                self.op = torch.minimum
            case "max(A, B)":
                self.op = torch.maximum
            case _:
                raise Exception("unsupported operation")

    def get_conds(self):
        return self.merge_conds(self.lhs.get_conds(), self.rhs.get_conds())

    def get_models(self):
        return self.lhs.get_models() | self.rhs.get_models()

    def get_preds(self):
        return {self} | self.lhs.get_preds() | self.rhs.get_preds()

    def predict_noise(self, x, timestep, model, conds, model_options, seed):
        lhs = self.lhs.predict_noise(x, timestep, model, conds, model_options, seed)
        rhs = self.rhs.predict_noise(x, timestep, model, conds, model_options, seed)
        return self.op(lhs, rhs)

    def reset_cache(self):
        # reset_cache is fast and idempotent so this is fine
        self.lhs.reset_cache()
        self.rhs.reset_cache()

class ScaledGuidancePredictor(NoisePredictor):
    """Implements A * scale + B"""
    INPUTS = {
        "required": {
            "guidance": ("PREDICTION",),
            "baseline": ("PREDICTION",),
            "guidance_scale": ("FLOAT", {
                "default": 6.0,
                "step": 0.1,
                "min": 0.0,
                "max": 100.0,
            }),
            "rescale": ("FLOAT", {
                "default": 0.0,
                "step": 0.1,
                "min": 0.0,
                "max": 100.0,
            }),
        }
    }

    def __init__(self, guidance, baseline, guidance_scale, rescale):
        self.lhs = guidance
        self.rhs = baseline
        self.scale = guidance_scale
        self.rescale = rescale

    def get_conds(self):
        return self.merge_conds(self.lhs.get_conds(), self.rhs.get_conds())

    def get_models(self):
        return self.lhs.get_models() | self.rhs.get_models()

    def get_preds(self):
        return {self} | self.lhs.get_preds() | self.rhs.get_preds()

    def predict_noise(self, x, sigma, model, conds, model_options, seed):
        lhs = self.lhs.predict_noise(x, sigma, model, conds, model_options, seed)
        rhs = self.rhs.predict_noise(x, sigma, model, conds, model_options, seed)

        sigma = sigma.view(sigma.shape[:1] + (1,) * (lhs.ndim - 1))
        x_scaled = x * (1. / (sigma.square() + 1.0))
        x_cfg = x_scaled - rhs - self.scale * lhs
        ro_pos = torch.std(x_scaled - (lhs + rhs), dim=(1, 2, 3), keepdim=True)
        ro_cfg = torch.std(x_cfg, dim=(1, 2, 3), keepdim=True)
        x_rescaled = x_cfg * (ro_pos / ro_cfg)
        x_final = torch.lerp(x_cfg, x_rescaled, self.rescale)

        return x_scaled - x_final

    def reset_cache(self):
        # reset_cache is fast and idempotent so this is fine
        self.lhs.reset_cache()
        self.rhs.reset_cache()

class AvoidErasePredictor(NoisePredictor):
    """Implements A - ((A proj B) * avoid_scale) - (B * erase_scale)"""
    INPUTS = {
        "required": {
            "guidance": ("PREDICTION",),
            "avoid_and_erase": ("PREDICTION",),
            "avoid_scale": ("FLOAT", {
                "default": 1.0,
                "step": 0.01,
                "min": 0.0,
                "max": 100.0,
            }),
            "erase_scale": ("FLOAT", {
                "default": 0.1,
                "step": 0.01,
                "min": 0.0,
                "max": 100.0,
            }),
        }
    }

    def __init__(self, guidance, avoid_and_erase, avoid_scale, erase_scale):
        self.lhs = guidance
        self.rhs = avoid_and_erase
        self.avoid_scale = avoid_scale
        self.erase_scale = erase_scale

    def get_conds(self):
        return self.merge_conds(self.lhs.get_conds(), self.rhs.get_conds())

    def get_models(self):
        return self.lhs.get_models() | self.rhs.get_models()

    def get_preds(self):
        return {self} | self.lhs.get_preds() | self.rhs.get_preds()

    def predict_noise(self, x, timestep, model, conds, model_options, seed):
        lhs = self.lhs.predict_noise(x, timestep, model, conds, model_options, seed)
        rhs = self.rhs.predict_noise(x, timestep, model, conds, model_options, seed)
        return lhs - (proj(lhs, rhs) * self.avoid_scale) - (rhs * self.erase_scale)

    def reset_cache(self):
        # reset_cache is fast and idempotent so this is fine
        self.lhs.reset_cache()
        self.rhs.reset_cache()

def dot(a, b):
    return (a * b).sum(dim=(1, 2, 3), keepdims=True)

def proj(a, b):
    a_dot_b = dot(a, b)
    b_dot_b = dot(b, b)
    divisor = torch.where(
        b_dot_b != 0,
        b_dot_b,
        torch.ones_like(b_dot_b)
    )

    return b * (a_dot_b / divisor)

def oproj(a, b):
    return a - proj(a, b)

class ScalePredictor(NoisePredictor):
    INPUTS = {
        "required": {
            "prediction": ("PREDICTION",),
            "scale": ("FLOAT", {
                "default": 1.0,
                "step": 0.01,
                "min": -100.0,
                "max": 100.0,
            })
        }
    }

    def __init__(self, prediction, scale):
        self.inner = prediction
        self.scale = scale

    def get_conds(self):
        return self.inner.get_conds()

    def get_models(self):
        return self.inner.get_models()

    def get_preds(self):
        return {self} | self.inner.get_preds()

    def predict_noise(self, x, timestep, model, conds, model_options, seed):
        return self.inner.predict_noise(x, timestep, model, conds, model_options, seed) * self.scale

    def reset_cache(self):
        self.inner.reset_cache()

class CFGPredictor(CachingNoisePredictor):
    INPUTS = {
        "required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "cfg_scale": ("FLOAT", {
                "default": 6.0,
                "min": 1.0,
                "max": 100.0,
                "step": 0.5,
            })
        }
    }

    def __init__(self, positive, negative, cfg_scale):
        self.positive = comfy.sample.convert_cond(positive)
        self.negative = comfy.sample.convert_cond(negative)
        self.cfg_scale = cfg_scale

    def get_conds(self):
        return {
            "positive": self.positive,
            "negative": self.negative,
        }

    def get_models(self):
        return self.get_models_from_conds()

    def n_samples(self):
        return 2

    def predict_noise_uncached(self, x, timestep, model, conds, model_options, seed):
        cond, uncond = comfy.samplers.calc_cond_uncond_batch(
            model,
            conds["positive"],
            conds["negative"],
            x,
            timestep,
            model_options
        )

        return uncond + (cond - uncond) * self.cfg_scale

class PerpNegPredictor(CachingNoisePredictor):
    INPUTS = {
        "required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "empty": ("CONDITIONING",),
            "cfg_scale": ("FLOAT", {
                "default": 6.0,
                "min": 1.0,
                "max": 100.0,
                "step": 0.5,
            }),
            "neg_scale": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 0.05,
            })
        }
    }

    def __init__(self, positive, negative, empty, cfg_scale, neg_scale):
        self.positive = comfy.sample.convert_cond(positive)
        self.negative = comfy.sample.convert_cond(negative)
        self.empty = comfy.sample.convert_cond(empty)
        self.cfg_scale = cfg_scale
        self.neg_scale = neg_scale

    def get_conds(self):
        return {
            "positive": self.positive,
            "negative": self.negative,
            "empty": self.empty,
        }

    def get_models(self):
        return self.get_models_from_conds()

    def n_samples(self):
        return 3

    def predict_noise_uncached(self, x, timestep, model, conds, model_options, seed):
        cond, uncond = comfy.samplers.calc_cond_uncond_batch(
            model,
            conds["positive"],
            conds["negative"],
            x,
            timestep,
            model_options
        )

        empty, _ = comfy.samplers.calc_cond_uncond_batch(
            model,
            conds["empty"],
            None,
            x,
            timestep,
            model_options
        )

        positive = cond - empty
        negative = uncond - empty
        perp_neg = oproj(negative, positive) * self.neg_scale
        return empty + (positive - perp_neg) * self.cfg_scale

NODE_CLASS_MAPPINGS = {
    "SamplerCustomPrediction": SamplerCustomPrediction,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerCustomPrediction": "Sample Predictions",
}

def make_node(predictor, display_name, class_name=None, category="sampling/prediction"):
    if class_name is None:
        class_name = predictor.__name__[:-2] + "ion" # Predictor -> Prediction

    cls = type(class_name, (), {
        "FUNCTION": "get_predictor",
        "CATEGORY": category,
        "INPUT_TYPES": classmethod(lambda cls: predictor.INPUTS),
        "RETURN_TYPES": tuple(predictor.OUTPUTS.values()),
        "RETURN_NAMES": tuple(predictor.OUTPUTS.keys()),
        "get_predictor": lambda self, **kwargs: (predictor(**kwargs),),
    })

    setattr(sys.modules[__name__], class_name, cls)
    NODE_CLASS_MAPPINGS[class_name] = cls
    NODE_DISPLAY_NAME_MAPPINGS[class_name] = display_name

make_node(ConditionedPredictor, "Conditioned Prediction")
make_node(CombinePredictor, "Combine Predictions", class_name="CombinePredictions")
make_node(ScaledGuidancePredictor, "Scaled Guidance Prediction")
make_node(AvoidErasePredictor, "Avoid and Erase Prediction")
make_node(ScalePredictor, "Scale Prediction")
make_node(CFGPredictor, "CFG Prediction")
make_node(PerpNegPredictor, "Perp-Neg Prediction")
