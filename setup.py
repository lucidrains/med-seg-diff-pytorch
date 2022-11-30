from setuptools import setup, find_packages

setup(
  name = 'med-seg-diff-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.4',
  license='MIT',
  description = 'MedSegDiff - SOTA medical image segmentation - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/med-seg-diff-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'denoising diffusion',
    'medical segmentation'
  ],
  install_requires=[
    'einops',
    'torch',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
