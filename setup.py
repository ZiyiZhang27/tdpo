from setuptools import setup

setup(
    name="tdpo-pytorch",
    version="0.0.1",
    packages=["tdpo_pytorch"],
    python_requires=">=3.10",
    install_requires=[
        "ml-collections",
        "absl-py",
        "diffusers[torch]==0.17.1",
        "accelerate==0.17",
        "wandb",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "inflect==6.0.4",
        "pydantic==1.10.9",
        "transformers==4.30.2",
        "opencv-python",
        "hpsv2",
        "image-reward",
        "clip",
    ],
)
