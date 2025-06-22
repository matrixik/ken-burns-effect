# PyTorch 1.3.1 → 2.7.1 Migration Summary

This document summarizes all the changes made to update the codebase from PyTorch 1.3.1 to 2.7.1.

## Major Changes Applied

### 1. PyTorch API Updates

#### Model Loading Security (`torch.load`)
- **Issue**: PyTorch 2.6+ changed default `weights_only` parameter from `False` to `True`
- **Solution**: Added explicit `weights_only` parameter to all `torch.load` calls
- **Files changed**: `utils/utils.py`

```python
# Before
checkpoint = torch.load(model)

# After 
checkpoint = torch.load(model, weights_only=False)  # For full checkpoint
state_dict = torch.load(model, weights_only=True)   # For state dict only
```

#### Pretrained Model Loading
- **Issue**: `pretrained=True` deprecated in favor of explicit weights parameter
- **Solution**: Updated model instantiation to use new weights format
- **Files changed**: `utils/utils.py`, `utils/pipeline.py`

```python
# Before
vgg_model = models.vgg16(pretrained=True)
maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# After
vgg_model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='MaskRCNN_ResNet50_FPN_Weights.COCO_V1')
```

#### Variable Deprecation
- **Issue**: `torch.autograd.Variable` deprecated - tensors track gradients directly
- **Solution**: Removed Variable wrapper usage
- **Files changed**: `utils/utils.py`, `utils/partial_conv.py`

```python
# Before
batch -= Variable(mean)
batch = torch.div(batch, Variable(std))

# After
batch -= mean
batch = torch.div(batch, std)
```

#### Tensor Type Operations
- **Issue**: `.type(torch.FloatTensor)` deprecated
- **Solution**: Use `.float()` method instead
- **Files changed**: `utils/utils.py`, `utils/common.py`

```python
# Before
tensor.type(torch.FloatTensor)
torch.FloatTensor([1, 2, 3]).cuda()

# After
tensor.float()
torch.tensor([1, 2, 3], dtype=torch.float32, device='cuda')
```

### 2. CuPy API Updates

#### Memoize Function Location
- **Issue**: `cupy.util.memoize` moved to `cupy.memoize`
- **Solution**: Updated import path
- **Files changed**: `utils/common.py`

```python
# Before
@cupy.util.memoize(for_each_device=True)

# After
@cupy.memoize(for_each_device=True)
```

### 3. CUDA Stream Handling
- **Issue**: `.cuda_stream` attribute removed from stream objects
- **Solution**: Use stream object directly
- **Files changed**: `utils/common.py`

```python
# Before
ptr = torch.cuda.current_stream().cuda_stream

# After
ptr = torch.cuda.current_stream()
```

### 4. Configuration Improvements

#### CUDA_HOME Environment
- **Issue**: Hardcoded CUDA paths not portable
- **Solution**: Made CUDA_HOME detection more flexible
- **Files changed**: `kbe.py`, `train.py`

```python
# Before
os.environ['CUDA_HOME'] = '/opt/cuda/cuda-10.1'

# After
if 'CUDA_HOME' not in os.environ:
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
```

#### Helper Math Path
- **Issue**: Hardcoded absolute path not portable
- **Solution**: Use relative path based on script location
- **Files changed**: `utils/common.py`

```python
# Before
path_to_math_helper = '/home/s182169/Master_thesis/GAN-Burns-effect/utils/helper_math.h'

# After
path_to_math_helper = os.path.join(os.path.dirname(__file__), 'helper_math.h')
```

### 5. Python Code Quality

#### Regex String Escaping
- **Issue**: Invalid escape sequences in regex patterns
- **Solution**: Use raw strings for regex patterns
- **Files changed**: `utils/common.py`

```python
# Before
re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

# After
re.search(r'(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)
```

### 6. Dependencies Update

#### Requirements.txt
- **Updated**: All packages to compatible versions
- **Files changed**: `requirements.txt`

```
torch==2.7.1
torchvision==0.22.1
kornia==0.8.1
cupy-cuda12x>=13.4.0  # Use cupy-cuda11x for CUDA 11.x
h5py
opencv-python
dill
moviepy
```

## Performance Optimizations

#### CuDNN Backend
- **Added**: `torch.backends.cudnn.benchmark = True` for consistent input sizes
- **Files changed**: `kbe.py`, `train.py`

## Testing

- Created `test_pytorch_compatibility.py` to verify all changes work correctly
- All tests pass with PyTorch 2.7.1 and CuPy 13.4.1

## Backward Compatibility

- All functionality preserved
- API changes are internal - external usage remains the same
- Model weights and checkpoints remain compatible

## Notes

- Code now requires PyTorch 2.6+ due to `weights_only` parameter usage
- CuPy 13.4+ required for Python 3.12 compatibility
- CUDA 12.x recommended (use `cupy-cuda11x` for CUDA 11.x systems)
- All original paper functionality maintained