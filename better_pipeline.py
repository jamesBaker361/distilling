
from diffusers import StableDiffusionPipeline

class BetterPipeline(StableDiffusionPipeline):
    def run_safety_checker(self, image, device, dtype):
        return image, None