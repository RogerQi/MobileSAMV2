# MobileSAMV2

A copy of MobileSAMv2 [implementation](https://github.com/ChaoningZhang/MobileSAM/tree/c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed/MobileSAMv2) for easy git submoduling.

## Usage

The models can be conveniently used as a torch.hub module.

```bash
# This will install necessary dependencies
pip install git+https://github.com/RogerQi/MobileSAMV2
# This line would invoke torch.hub to automatically download weights and stuff
python hub_inference_example.py
```

**Disclaimer:** This codebase is based on the original MobileSAMV2 and is intended only to be used as a wrapper for convenience such as fast prototyping. I inherited the same Apache-2.0 LICENSE. If you intend to use it in your research/product, please refer to the original repository for more details and proper licensing.
