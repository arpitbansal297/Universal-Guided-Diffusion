"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import GPUtil
from torchvision import transforms, utils


from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSamplerWithGrad(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):

        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.model.module.num_timesteps, verbose=verbose)

        alphas_cumprod = self.model.module.alphas_cumprod
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.module.device)

        self.register_buffer('betas', to_torch(self.model.module.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.module.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    def sample(self,
               S,
               batch_size,
               shape,
               operated_image=None,
               operation=None,
               conditioning=None,
               eta=0.,
               temperature=1.,
               verbose=True,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               start_zt=None
               ):


        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        shape = (batch_size, C, H, W)
        cond = conditioning


        device = self.model.module.betas.device
        b = shape[0]

        if start_zt is None:
            img = torch.randn(shape, device=device)
            start_zt = img
        else:
            img = start_zt

        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        for param in self.model.module.first_stage_model.parameters():
            param.requires_grad = False



        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            b, *_, device = *img.shape, img.device

            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            beta_t = a_t / a_prev

            # num_step_length = len(operation.num_steps)
            # index_n = int(num_step_length * (ts[0] / self.num_timesteps))
            # num_steps = operation.num_steps[index_n]
            num_steps = operation.num_steps[0]

            loss = None
            _ = None

            operation_func = operation.operation_func
            other_guidance_func = operation.other_guidance_func
            criterion = operation.loss_func
            other_criterion = operation.other_criterion
            max_iters = operation.max_iters
            loss_cutoff = operation.loss_cutoff

            for j in range(num_steps):

                if operation.guidance_3:

                    torch.set_grad_enabled(True)
                    img_in = img.detach().requires_grad_(True)

                    if operation.original_guidance:
                        x_in = torch.cat([img_in] * 2)
                        t_in = torch.cat([ts] * 2)
                        c_in = torch.cat([unconditional_conditioning, cond])
                        e_t_uncond, e_t = self.model.module.apply_model(x_in, t_in, c_in).chunk(2)
                        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
                        # del x_in
                    else:
                        e_t = self.model.module.apply_model(img_in, ts, cond)

                    pred_x0 = (img_in - sqrt_one_minus_at * e_t) / a_t.sqrt()
                    recons_image = self.model.module.decode_first_stage_with_grad(pred_x0)

                    if other_guidance_func != None:
                        op_im = other_guidance_func(recons_image)
                    elif operation_func != None:
                        op_im = operation_func(recons_image)
                    else:
                        op_im = recons_image

                    if op_im is not None:
                        if other_criterion != None:
                            selected = -1 * other_criterion(op_im, operated_image)
                        else:
                            selected = -1 * criterion(op_im, operated_image)

                        print(ts)
                        print(selected)

                        grad = torch.autograd.grad(selected.sum(), img_in)[0]
                        grad = grad * operation.optim_guidance_3_wt

                        e_t = e_t - sqrt_one_minus_at * grad.detach()

                        img_in = img_in.requires_grad_(False)

                        if operation.print:
                            if j == 0:
                                temp = (recons_image + 1) * 0.5
                                utils.save_image(temp, f'{operation.folder}/img_at_{ts[0]}.png')

                        del img_in, pred_x0, recons_image, op_im, selected, grad
                        if operation.original_guidance:
                            del x_in

                    else:
                        e_t = e_t

                        img_in = img_in.requires_grad_(False)

                        if operation.print:
                            if j == 0:
                                temp = (recons_image + 1) * 0.5
                                utils.save_image(temp, f'{operation.folder}/img_at_{ts[0]}.png')

                        del img_in, pred_x0, recons_image, op_im
                        if operation.original_guidance:
                            del x_in


                    torch.set_grad_enabled(False)

                else:
                    if operation.original_guidance:
                        x_in = torch.cat([img] * 2)
                        t_in = torch.cat([ts] * 2)
                        c_in = torch.cat([unconditional_conditioning, cond])
                        e_t_uncond, e_t = self.model.module.apply_model(x_in, t_in, c_in).chunk(2)
                        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
                    else:
                        e_t = self.model.module.apply_model(img, ts, cond)

                with torch.no_grad():
                    # current prediction for x_0
                    pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()

                    # direction pointing to x_t
                    dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
                    noise = sigma_t * noise_like(img.shape, device, False) * temperature

                    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
                    img = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(img.shape, device, False)

                    del pred_x0, dir_xt, noise

            img = x_prev


        return img, start_zt


    def sample_seperate(self,
               S,
               batch_size,
               shape,
               operated_image=None,
               operation=None,
               conditioning=None,
               eta=0.,
               temperature=1.,
               verbose=True,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               ):


        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        shape = (batch_size, C, H, W)
        cond = conditioning


        device = self.model.module.betas.device
        b = shape[0]

        img = torch.randn(shape, device=device)

        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        for param in self.model.module.first_stage_model.parameters():
            param.requires_grad = False



        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            b, *_, device = *img.shape, img.device

            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            beta_t = a_t / a_prev

            # num_step_length = len(operation.num_steps)
            # index_n = int(num_step_length * (ts[0] / self.num_timesteps))
            # num_steps = operation.num_steps[index_n]
            num_steps = operation.num_steps[0]

            loss = None
            _ = None

            operation_func = operation.operation_func
            other_guidance_func = operation.other_guidance_func
            criterion = operation.loss_func
            other_criterion = operation.other_criterion
            max_iters = operation.max_iters
            loss_cutoff = operation.loss_cutoff

            for j in range(num_steps):

                if operation.guidance_3:

                    torch.set_grad_enabled(True)
                    img_in = img.detach().requires_grad_(True)

                    if operation.original_guidance:
                        x_in = torch.cat([img_in] * 2)
                        t_in = torch.cat([ts] * 2)
                        c_in = torch.cat([unconditional_conditioning, cond])
                        e_t_uncond, e_t_cond = self.model.module.apply_model(x_in, t_in, c_in).chunk(2)
                        e_t = e_t_uncond # + unconditional_guidance_scale * (e_t - e_t_uncond)
                        # del x_in
                    else:
                        e_t = self.model.module.apply_model(img_in, ts, cond)

                    pred_x0 = (img_in - sqrt_one_minus_at * e_t) / a_t.sqrt()
                    recons_image = self.model.module.decode_first_stage_with_grad(pred_x0)

                    if other_guidance_func != None:
                        op_im = other_guidance_func(recons_image)
                    elif operation_func != None:
                        op_im = operation_func(recons_image)
                    else:
                        op_im = recons_image

                    if op_im is not None:
                        if other_criterion != None:
                            selected = -1 * other_criterion(op_im, operated_image)
                        else:
                            selected = -1 * criterion(op_im, operated_image)

                        print(ts)
                        print(selected)

                        grad = torch.autograd.grad(selected.sum(), img_in)[0]
                        grad = grad * operation.optim_guidance_3_wt

                        e_t = e_t - sqrt_one_minus_at * grad.detach() + unconditional_guidance_scale * (e_t_cond - e_t_uncond)

                        img_in = img_in.requires_grad_(False)

                        if operation.print:
                            if j == 0:
                                temp = (recons_image + 1) * 0.5
                                utils.save_image(temp, f'{operation.folder}/img_at_{ts[0]}.png')

                        del img_in, pred_x0, recons_image, op_im, selected, grad, e_t_uncond
                        if operation.original_guidance:
                            del x_in

                    else:
                        e_t = e_t + unconditional_guidance_scale * (e_t_cond - e_t_uncond)

                        img_in = img_in.requires_grad_(False)

                        if operation.print:
                            if j == 0:
                                temp = (recons_image + 1) * 0.5
                                utils.save_image(temp, f'{operation.folder}/img_at_{ts[0]}.png')

                        del img_in, pred_x0, recons_image, op_im
                        if operation.original_guidance:
                            del x_in


                    torch.set_grad_enabled(False)

                else:
                    if operation.original_guidance:
                        x_in = torch.cat([img] * 2)
                        t_in = torch.cat([ts] * 2)
                        c_in = torch.cat([unconditional_conditioning, cond])
                        e_t_uncond, e_t = self.model.module.apply_model(x_in, t_in, c_in).chunk(2)
                        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
                    else:
                        e_t = self.model.module.apply_model(img, ts, cond)

                with torch.no_grad():
                    # current prediction for x_0
                    pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()

                    # direction pointing to x_t
                    dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
                    noise = sigma_t * noise_like(img.shape, device, False) * temperature

                    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
                    img = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(img.shape, device, False)

                    del pred_x0, dir_xt, noise

            img = x_prev


        return img


    def ddim_sampling(self, cond, shape,operated_image=None, operation=None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.module.betas.device
        b = shape[0]

        img = torch.randn(shape, device=device)

        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        for param in self.model.module.first_stage_model.parameters():
            param.requires_grad = False

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            b, *_, device = *img.shape, img.device

            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            operation_func = operation.operation_func
            other_guidance_func = operation.other_guidance_func
            criterion = operation.loss_func
            other_criterion = operation.other_criterion
            max_iters = operation.max_iters
            loss_cutoff = operation.loss_cutoff

            if operation.guidance_3:

                print(GPUtil.showUtilization())

                torch.set_grad_enabled(True)
                img_in = img.detach().requires_grad_(True)
                #
                # optimizer = torch.optim.SGD([img_in], lr=1)
                # optimizer.zero_grad()

                if operation.original_guidance:
                    x_in = torch.cat([img_in] * 2)
                    t_in = torch.cat([ts] * 2)
                    c_in = torch.cat([unconditional_conditioning, cond])
                    e_t_uncond, e_t = self.model.module.apply_model(x_in, t_in, c_in).chunk(2)
                    e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
                    # del x_in
                else:
                    e_t = self.model.module.apply_model(img_in, ts, cond)

                pred_x0 = (img_in - sqrt_one_minus_at * e_t) / a_t.sqrt()
                recons_image = self.model.module.decode_first_stage_with_grad(pred_x0)


                if other_guidance_func != None:
                    op_im = other_guidance_func(recons_image)
                elif operation_func != None:
                    op_im = operation_func(recons_image)
                else:
                    op_im = recons_image

                if other_criterion != None:
                    selected = -1 * other_criterion(op_im, operated_image)
                else:
                    selected = -1 * criterion(op_im, operated_image)

                print(ts)
                print(selected)

                grad = torch.autograd.grad(selected.sum(), img_in)[0]
                grad = grad * operation.optim_guidance_3_wt

                # selected.sum().backward()
                # optimizer.step()
                # optimizer.zero_grad()
                # grad = (img - img_in) * operation.optim_guidance_3_wt

                e_t = e_t - sqrt_one_minus_at * grad.detach()

                img_in = img_in.requires_grad_(False)

                del img_in, pred_x0, recons_image, op_im, selected, grad
                if operation.original_guidance:
                    del x_in
                torch.set_grad_enabled(False)
                print("Done ?")

                print(GPUtil.showUtilization())

            else:
                if operation.original_guidance:
                    x_in = torch.cat([img] * 2)
                    t_in = torch.cat([ts] * 2)
                    c_in = torch.cat([unconditional_conditioning, cond])
                    e_t_uncond, e_t = self.model.module.apply_model(x_in, t_in, c_in).chunk(2)
                    e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
                else:
                    e_t = self.model.module.apply_model(img, ts, cond)

            with torch.no_grad():
            # current prediction for x_0
                pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()

                # direction pointing to x_t
                dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
                noise = sigma_t * noise_like(img.shape, device, False) * temperature

                x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
                img = x_prev

                del pred_x0, dir_xt, noise, x_prev


        return img


    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev


    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)


    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec