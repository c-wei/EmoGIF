import math
import numpy as np
import torch
import enum
from tqdm.auto import tqdm

class ModelVarianceType(enum.Enum):
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()

    # TODO: 
    LEARNED = enum.auto()

class ModelMeanType(enum.Enum):
    X_PREV = enum.auto()
    X_0 = enum.auto()
    EPSILON = enum.auto()

def warmup_beta(beta_start, beta_end, num_diff_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diff_timesteps, dtype=np.float64)
    warmup_time = int(num_diff_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas

def get_beta_schedule(schedule_name, num_diff_timesteps):
    if schedule_name == 'linear':
        scale = 1000/num_diff_timesteps
        betas = np.linspace(scale * 0.0001, scale * 0.2, num_diff_timesteps, dtype=np.float64)
        
        assert betas.shape == (num_diff_timesteps,)
        return betas
    else:
        raise NotImplementedError(f"uknown beta schedule: {schedule_name}")
    

def extract_into_tensor(arr, timesteps, broadcast_shape):
    """extract 1-D array into a [batch_size, 1, ...] shaped tensor"""
    res = torch.from_nump(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[...,None]

    return res + torch.zeros(broadcast_shape, device=timesteps.device)


class GaussianDiffusion:
    def __init__(
            self,
            *,
            betas,
            loss_type,
            model_variance_type,
            model_mean_type
    ):
        self.model_mean_type = model_mean_type
        self.model_variance_type = model_variance_type
        self.loss_type = loss_type
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1D"
        assert (betas > 0).all() and (betas<=1).all()

        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        self.pos_mean_x0_coeff = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.pos_mean_xt_coeff = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))
        self.pos_variance = (betas * (1.0-self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.pos_log_variance_clipped = np.log(np.append(self.variance[1], self.variance[1:])) if len(self.variance) > 1 else np.array([])

    '''Forward Diffusion Process'''

    def q_mean_variance(self, x_0, t):
        mean = extract_into_tensor(np.sqrt(self.alphas_cumprod), t, x_0.shape) * x_0
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_0.shape)
        log_variance = extract_into_tensor(np.log(1.0 - self.alphas_cumprod), t, x_0.shape)
        return mean, variance, log_variance


    def q_sample(self, x_0, t, noise=None):
        """"Diffusion step"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        assert noise.shape == x_0.shape
        return (extract_into_tensor(np.sqrt(self.alphas_cumprod), t, x_0.shape) * x_0 + 
                    extract_into_tensor(np.sqrt(1.0 - self.alphas_cumprod), t, x_0.shape) * noise)
        

    def q_pos_mean_variance(self, x_0, x_t, t):
        """Compute mean & variance q(x_t | x_0) ~ p(x_t-1|x_t)"""
        assert x_0.shape == x_t.shape
        pos_mean = (extract_into_tensor(self.pos_mean_x0_coeff, t, x_t.shape) * x_0 +
                    extract_into_tensor(self.pos_mean_xt_coeff, t, x_t.shape) * x_t)
        
        pos_variance = extract_into_tensor(self.pos_variance, t, x_t.shape)
        pos_log_variance_clipped = extract_into_tensor(self.pos_log_variance_clipped, t, x_t.shape)

        assert(
            pos_mean.shape[0] == pos_variance.shape[0] == pos_log_variance_clipped.shape[0] == x_0.shape[0]
        )
        return pos_mean, pos_variance, pos_log_variance_clipped


    '''Reverse Diffusion Process'''

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """Calculate p(x_{t-1}|x_t)"""

        if model_kwargs is None:
            model_kwargs = {}

        B,C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x,t,**model_kwargs)
        
        if isinstance(model_output, tuple):
            model_output, extra_output = model_output
        else:
            extra_output = None
        
        # TODO: implement a learned variance model

        model_variance, model_log_variance = {
            ModelVarianceType.FIXED_LARGE:(
                np.append(self.pos_variance[1], self.betas[1:]),
                np.log(np.append(self.pos_variance[1], self.betas[1:])),
            ),
            ModelVarianceType.FIXED_SMALL: (
                self.pos_variance,
                self.pos_log_variance_clipped,
            ),
        }[self.model_var_type]
        model_variance = extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = extract_into_tensor(model_log_variance, t, x.shape)

        def process_x0(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1,1)
            return x
        
        if self.model_mean_type == ModelMeanType.X_0:
            pred_x0 = process_x0(model_output)
        else:
            pred_x0 = process_x0(
                self.x0_from_epsilon(x, t, model_output)
            )

        model_mean,_,_ = self.q_pos_mean_variance(x_0=pred_x0, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_x0.shape == x.shape

        return{
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': model_log_variance,
            'pred_x0': pred_x0,
            'extra': extra_output,
        }

    def x0_from_epsilon(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return(
            extract_into_tensor(np.sqrt(1.0 / self.alphas_cumprod), t, x_t.shape) * x_t - 
            extract_into_tensor(np.sqrt(1.0 / self.alphas_cumprod - 1.0), t, x_t.shape) * eps
        )
    def eps_from_x0(self, x_t, t, pred_x0):
        return(
            extract_into_tensor(np.sqrt(1.0 / self.alphas_cumprod), t, x_t.shape) * x_t - pred_x0 /
            extract_into_tensor(np.sqrt(1.0 / self.alphas_cumprod - 1.0), t, x_t.shape)
        )

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        '''For score-based models, Song et al(2020)'''
        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self.eps_from_x0(x, t, p_mean_var['pred_x0'])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        out['pred_x0'] = self.x0_from_epsilon(x, t, eps)
        out['mean'],_,_ = self.q_pos_mean_variance(x_0 = out['pred_x0'], x_t=x, t=t)

        return out
    

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        '''sample p(x_{t-1}|x_t)'''

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        noise = torch.randn_like(x)
        mask = ((t!=0).float().view(-1, *([1] * (len(x.shape) - 1))))

        if cond_fn is not None:
            out['mean'] = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out['mean'] + mask * torch.exp(0.5 * out['log_variance']) * noise
        return {'sample': sample, 'pred_x0': out['pred_x0']}


    def p_generate_samples(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
    ):
        '''Generate all samples'''
        final = None
        for sample in self.p_generate_sample_t(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
        ):
            final = sample
        return final['sample']

    def p_generate_sample_t(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
    ):
        '''Generate sample from time t'''
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple,list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs
                )

                yield out
                img = out['sample']

        
