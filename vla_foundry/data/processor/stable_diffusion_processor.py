import torch
from torchvision import transforms
from transformers import CLIPTokenizer


class StableDiffusionProcessor:
    def __init__(self, image_size=512, max_length=64, tokenizer_name="openai/clip-vit-base-patch32"):
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        self.image_size = image_size
        self.max_length = max_length
        self.image_token_id = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __call__(self, **sample):
        # Process text
        text_inputs = self.tokenizer(
            sample["text"], padding="max_length", max_length=self.max_length + 1, truncation=True, return_tensors="pt"
        )

        # Process image
        pixel_values = torch.stack([self.transform(img) for img in sample["images"]])

        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": pixel_values,
        }
