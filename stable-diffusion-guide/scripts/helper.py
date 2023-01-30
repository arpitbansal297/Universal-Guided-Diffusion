class OptimizerDetails:
    def __init__(self):
        self.num_steps = None
        self.operation_func = None
        self.optimizer = None # handle it on string level
        self.lr = None
        self.loss_func = None
        self.max_iters = 0
        self.loss_cutoff = None
        self.lr_scheduler = None
        self.warm_start = None
        self.old_img = None
        self.fact = 0.5
        self.print = False
        self.print_every = None
        self.folder = None
        self.tv_loss = None
        self.guidance_3 = False
        self.guidance_2 = False
        self.Aug = None
        self.optim_guidance_3_wt = 0
        self.optim_guidance_3_iters = 1
        self.optim_unscaled_guidance_3 = False
        self.mask_type = 3
        self.other_guidance_func = None
        self.other_criterion = None
        self.original_guidance = False
        self.guidance_mask = None
        self.do_guidance_3_norm = False
        self.sampling_type = None
        self.loss_save = None
        self.ddpm = False

def get_face_text(text_type):

    if text_type == 1:
        prompt = "Headshot of a person with blonde hair"
    elif text_type == 2:
        prompt = "A headshot of person with space background"
    elif text_type == 3:
        prompt = "Headshot of a person with blue hair"
    elif text_type == 4:
        prompt = "Headshot of a sad person with blonde hair"
    elif text_type == 5:
        prompt = "Headshot of a person with blonde hair with space background"
    elif text_type == 6:
        prompt = "A happy person smiling and riding a horse"
    elif text_type == 7:
        prompt = "A happy person smiling and riding a car"
    elif text_type == 8:
        prompt = "A headshot of a woman with tattoo"
    elif text_type == 9:
        prompt = "A headshot of a woman as astronaut"
    elif text_type == 10:
        prompt = "A headshot of a 60 year old woman"
    elif text_type == 11:
        prompt = "A headshot of a woman looking like a lara croft"
    elif text_type == 12:
        prompt = "A headshot of a blonde woman as a knight"
    elif text_type == 13:
        prompt = "A headshot of a woman muppet"
    elif text_type == 14:
        prompt = "A headshot of a blonde woman muppet"
    elif text_type == 15:
        prompt = "A headshot of a woman made of marble"
    elif text_type == 16:
        prompt = "A headshot of a blonde woman made of marble"
    elif text_type == 17:
        prompt = "A headshot of a blonde woman as a sketch"
    elif text_type == 18:
        prompt = "A headshot of a blonde woman as a japanese painting"
    elif text_type == 19:
        prompt = "A headshot of a blonde woman as oil painting"
    else:
        prompt = ""

    return prompt


def get_seg_text(text_type):

    if text_type == 1:
        prompt = ""
    elif text_type == 2:
        prompt = " in space"
    elif text_type == 3:
        prompt = " on snow"
    elif text_type == 4:
        prompt = " under water"
    elif text_type == 5:
        prompt = " on beach"
    elif text_type == 6:
        prompt = " on times square"
    elif text_type == 7:
        prompt = " on moon"
    elif text_type == 8:
        prompt = " in forest"
    elif text_type == 9:
        prompt = " on mars"
    elif text_type == 10:
        prompt = " as a sketch"
    elif text_type == 11:
        prompt = " as an oil painting"
    elif text_type == 12:
        prompt = " as a japanese painting"

    else:
        prompt = ""

    return prompt