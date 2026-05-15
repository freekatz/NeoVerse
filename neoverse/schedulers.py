import torch

from wan.utils.fm_solvers import get_sampling_sigmas


class FlowMatchScheduler:
    """NeoVerse scheduler wrapper backed by Wan's official sigma schedule.

    Wan's official schedulers are inference-oriented and do not expose the small
    training helpers used by NeoVerse. This wrapper keeps the NeoVerse call
    surface while deriving inference sigmas from `wan.utils.fm_solvers`.
    """

    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.0,
        inverse_timesteps=False,
        extra_one_step=True,
        reverse_sigmas=False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None):
        if shift is not None:
            self.shift = shift
        if training:
            sigmas = torch.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1)[:-1]
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        elif denoising_strength == 1.0 and self.sigma_max == 1.0 and self.sigma_min == 0.0 and not self.inverse_timesteps and not self.reverse_sigmas:
            sigmas = torch.as_tensor(get_sampling_sigmas(num_inference_steps, self.shift), dtype=torch.float32)
        else:
            sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
            count = num_inference_steps + 1 if self.extra_one_step else num_inference_steps
            sigmas = torch.linspace(sigma_start, self.sigma_min, count)
            if self.extra_one_step:
                sigmas = sigmas[:-1]
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        if self.inverse_timesteps:
            sigmas = torch.flip(sigmas, dims=[0])
        if self.reverse_sigmas:
            sigmas = 1 - sigmas
        self.sigmas = sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        self.training = bool(training)
        if self.training:
            self.set_training_weight(num_inference_steps)

    def set_training_weight(self, num_inference_steps):
        x = self.timesteps
        y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
        y_shifted = y - y.min()
        self.linear_timesteps_weights = y_shifted * (num_inference_steps / y_shifted.sum())

    def _timestep_id(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        return torch.argmin((self.timesteps - timestep).abs())

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        timestep_id = self._timestep_id(timestep)
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_next = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_next = self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_next - sigma)

    def return_to_timestep(self, timestep, sample, sample_stablized):
        sigma = self.sigmas[self._timestep_id(timestep)]
        return (sample - sample_stablized) / sigma

    def add_noise(self, original_samples, noise, timestep):
        sigma = self.sigmas[self._timestep_id(timestep)]
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample, noise, timestep):
        return noise - sample

    def training_weight(self, timestep):
        return self.linear_timesteps_weights[self._timestep_id(timestep)]
