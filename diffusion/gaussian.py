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

def calc_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor=obj
            break
    logvar1, logvar2 = [x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor) for x in (logvar1, logvar2)]

    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1-logvar2) + ((mean1-mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


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
        }[self.model_variance_type]
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
        
    # DDIM Implementation
    def p_ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """DDIM Implementation to sample x_{t-1}"""
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs
        )

        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        eps = self.eps_from_x0(x, t, out['pred_x0'])
        a_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)
        a_bar_prev = extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sig = (
            eta * torch.sqrt((1 - a_bar_prev) / (1 - a_bar)) * torch.sqrt(1 - a_bar / a_bar_prev)
        )

        noise = torch.randn_like(x)
        pred_mean = ( out['pred_x0'] * torch.sqrt(a_bar_prev) + torch.sqrt(1-a_bar_prev - sig ** 2) * eps)
        mask = ((t != 0).float().view(-1, *([1] * len(x.shape) -1)))

        sample = pred_mean + mask * sig * noise
        return {'sample': sample, 'pred_x0': out['pred_x0']}
    
    def q_ddim_reverse(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """Sample x_{t+1} to invert deterministic DDIM"""
        eta=0.0
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        eps=(extract_into_tensor(np.sqrt(1.0/self.alphas_cumprod), t, x.shape) * x - out['pred_x0']) / extract_into_tensor(np.sqrt(1.0/self.alphas_cumprod - 1), t, x.shape)
        a_bar_next = extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        
        pred_mean=out['pred_x0'] * torch.sqrt(a_bar_next) + torch.sqrt(1-a_bar_next) * eps
        return {'sample': pred_mean, 'pred_x0': out['pred_x0']}
    
    def p_ddim_sampler(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=True,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        eta=0.0
    ):
        final = None
        for sample in self.p_ddim_generate_sample(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            eta=eta,
        ):
            final = sample
        return final['sample']


    def p_ddim_generate_sample(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=True,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        eta=0.0
    ):
        if device is not None:
            device = next(model.paraeters()).device
        if noise is not None:
            img = noise
        
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_ddim_sampleddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )

                yield out
                img = out['sample']

    def var_lower_bound ( self, model, x_0, x_t, t, clip_denoised=True, model_kwargs=None):
        """Calculate per-step contribution to ELBO"""

        mean, _, log_var = self.q_pos_mean_variance(x_0=x_0, x_t=x_t, t=t)
        out = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = calc_kl(mean, log_var, out['mean'], out['log_variance'])
        kl = (kl.mean(dim=list(range(1,(len(kl.shape)))))) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(x_0, means=out['mean'], log_scales=0.5 * out['log_variance'])
        decoder_nll = (decoder_nll.mean(dim=list(range(1,(len(decoder_nll.shape)))))) / np.log(2.0)

        output = torch.where((t==0), decoder_nll, kl)
        return {'output': output, 'pred_x0': out['pred_x0']}

    def loss(
        self,
        model,
        x_0,
        t,
        model_kwargs=None,
        noise=None,
    ):
        """Loss type: rescaledMSE"""

        if model_kwargs is None:
            model_kwargs={}
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise=noise)

        terms = {}
        output = model(x_t, t, **model_kwargs)

        if self.model_variance_type in [
                ModelVarianceType.LEARNED,
        ]:
            B, C = x_t.shape[:2]
            assert output.shape == (B,C * 2, *x_t.shape[2:])
            output, var_vals = torch.split(output, C, dim=1)

            detached_out = torch.cat([output.detach(), var_vals], dim=1)
            terms['vb'] = self.var_lower_bound(
                model=lambda *args, r=detached_out: r,
                x_0=x_0,
                x_t=x_t,
                t=t,
                clip_denoised=False,
            )['output']
            terms['vb'] *= self.num_timesteps/1000.0

        target = {
            ModelMeanType.X_PREV: self.q_pos_mean_variance(
                x_start=x_0, x_t=x_t, t=t
            )[0],
            ModelMeanType.X_0: x_0,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert output.shape == target.shape == x_0.shape
        terms["mse"] = (((target - output) ** 2).mean(dim=list(range(1,(len(((target - output) ** 2).shape)))))) 
        
        # Optional VB loss
        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]
        
        return terms
                

    def prior_kl(self, x_0):
        """Calculate KL[q(x_T|x_0) || p(x_T)]"""
        batch_size = x_0.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_0.device)
        mean, _, log_var = self.q_mean_variance(x_0,t)
        kl = calc_kl(mean1=mean, logvar1=log_var, mean2=0.0, logvar=0.0)
        return (kl.mean(dim=list(range(1,(len(kl.shape)))))) / np.log(2.0)
    

    def calc_kl_loop(self, model, x_0, clip_denoised=True, model_kwargs=None):
        """Loop backward over timestep & collect per-step KL contributions, error between p(x_0) and x_0, and eps v. true noise"""
        device = x_0.device
        batch_size = x_0.shape[0]
        var_bound=[]
        x0_mse=[]
        mse=[]

        for timestep in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([timestep] * batch_size, device=device)
            noise = torch.randn_like(x_0)
            x_t = self.q_sample(x_0=x_0, t=t_batch, noise=noise)
            with torch.no_grad():
                out = self.var_lower_bound(
                    model,
                    x_0=x_0,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            var_bound.append(out['output'])
            x0_mse.append(((out['pred_x0']-x_0).mean(dim=list(range(1,(len((out['pred_x0']-x_0).shape))))))**2)
            eps = self.eps_from_x0(x_t, t_batch, out['pred_x0'])
            mse.append(((eps-noise).mean(dim=list(range(1,(len((eps-noise).shape))))))**2)


        var_bound = torch.stack(var_bound, dim=1)
        x0_mse = torch.stack(x0_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_kl = self.prior_kl(x_0)
        total_kl = var_bound.sum(dim=1) + prior_kl
        return{
            'total_bound': total_kl,
            'prior_bound': prior_kl,
            'var_bound': var_bound,
            'x0_mse': x0_mse,
            'mse': mse,
        }

