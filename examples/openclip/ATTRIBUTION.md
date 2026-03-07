# OpenCLIP Example Fixture Attribution

This example bundles resized local derivative images for offline execution.
All bundled files are 224x224 PNG images generated from Hugging Face datasets.
MNIST and Fashion-MNIST source images are grayscale and are converted to RGB
during fixture generation.

## Source Datasets

- `ylecun/mnist`
  - URL: <https://huggingface.co/datasets/ylecun/mnist>
  - Split: `test`
  - Indexes used: `0-9`
  - License listed on dataset card: MIT
- `zalando-datasets/fashion_mnist`
  - URL: <https://huggingface.co/datasets/zalando-datasets/fashion_mnist>
  - Split: `test`
  - Indexes used: `0-9`
  - License listed on dataset card: MIT
- `AI-Lab-Makerere/beans`
  - URL: <https://huggingface.co/datasets/AI-Lab-Makerere/beans>
  - Split: `test`
  - Indexes used: `0-9`
  - License listed on dataset card: MIT

## Notes

- These files are resized derivative assets included only for the runnable
  OpenCLIP example and documentation.
- Source license metadata above was retrieved on March 7, 2026.
