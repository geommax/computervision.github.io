


GitHub Knowledge Sharing Page အတွက် **Module 4 ၏ နောက်ဆုံးအခန်းဖြစ်သော Chapter 4.4** ကို ရေးသားပေးလိုက်ပါတယ်။ ဒီအခန်းမှာ Computer Vision AI တွေကို လက်တွေ့ရေးသားတဲ့အခါ Beginner တွေ အများဆုံး မျက်စိလည်၊ မှားယွင်းတတ်တဲ့ **Dimension, Channel ဖွဲ့စည်းပုံတွေနဲ့ PyTorch Data အမျိုးအစားတွေ** ကို သေချာကွဲပြားသွားအောင် အသေးစိတ် ရှင်းပြပေးထားပါတယ်။

---

# 📘 Chapter 4.4: Introduction to PyTorch Tensors (PyTorch Tensors များနှင့် မိတ်ဆက်ခြင်း)

အရင်အခန်းမှာ AI တွေ ဘယ်လိုသင်ယူလဲဆိုတဲ့ Calculus တွက်ချက်မှု (Gradient တွေ၊ Backpropagation တွေ) ကို လေ့လာခဲ့ပါတယ်။ တကယ့် လက်တွေ့ လုပ်ငန်းခွင်မှာ အဲ့ဒီ သင်္ချာတွေကို ကိုယ်တိုင်တွက်စရာ မလိုတော့ပါဘူး။ **PyTorch** က သင့်အတွက် အလိုအလျောက် အားလုံး တွက်ချက်ပေးမှာ ဖြစ်ပါတယ်။

ဒါပေမယ့် PyTorch ကို အသုံးပြုဖို့အတွက် သူ့ရဲ့ အသက်သွေးကြောဖြစ်တဲ့ **Tensor** ဆိုတဲ့ Data တည်ဆောက်ပုံကို သေချာနားလည်ဖို့ လိုအပ်ပါတယ်။

---

## 📦 ၁။ Tensor ဆိုတာ ဘာလဲ? NumPy နဲ့ ဘာကွာသလဲ?

Tensor ဆိုတာ တခြားမဟုတ်ပါဘူး၊ **ဂဏန်းတွေ စုစည်းထားတဲ့ ဇယားကွက် (Matrix) ကြီး** ပါပဲ။ အရင်အခန်းတွေက OpenCV မှာ သင်ခဲ့တဲ့ `Numpy Array` နဲ့ သဘောတရား အတိအကျ တူညီပါတယ်။

**ဒါဆို ဘာလို့ Numpy ကို ဆက်မသုံးဘဲ Tensor ကို သုံးရတာလဲ?**
1. **GPU အထောက်အပံ့ (GPU Acceleration):** Numpy ဟာ CPU ပေါ်မှာပဲ အလုပ်လုပ်ပါတယ်။ Tensor တွေကတော့ NVIDIA GPU တွေပေါ်ကို အလွယ်တကူ ရွှေ့ပြောင်းနိုင်ပြီး သန်းနဲ့ချီတဲ့ တွက်ချက်မှုတွေကို တစ်ပြိုင်နက်တည်း (Parallel) အလွန်မြန်ဆန်စွာ လုပ်ဆောင်နိုင်ပါတယ်။
2. **Autograd (Automatic Differentiation):** Tensor တွေဟာ သူ့အပေါ်မှာ ဘာသင်္ချာတွေ တွက်သွားလဲဆိုတာကို မှတ်သားထားနိုင်ပြီး၊ Backpropagation လုပ်တဲ့အခါ Gradient တွေကို အလိုအလျောက် တွက်ထုတ်ပေးနိုင်ပါတယ်။ (Numpy မှာ ဒီစွမ်းရည် မရှိပါဘူး)။

---

## 📏 ၂။ Dimensions ကို နားလည်ခြင်း (0D မှ 4D သို့)

PyTorch မှာ Tensor တစ်ခုဆောက်လိုက်ရင် သူ့ရဲ့ **Dimension (အတိုင်းအတာ/အလွှာ)** ဘယ်လောက်ရှိလဲ ဆိုတာကို အမြဲသတိပြုရပါတယ်။ Computer Vision မှာ အဓိကတွေ့ရမယ့် Dimension ပုံစံ ၅ မျိုး ရှိပါတယ်။

*   **0-Dimension (Scalar):** ဂဏန်းတစ်လုံးတည်း။ (ဥပမာ - `5`, `0.01`)။ Loss တန်ဖိုးတွေဟာ 0D ဖြစ်ပါတယ်။
*   **1-Dimension (Vector):** ဂဏန်းတွေကို တန်းစီထားတဲ့ List တစ်ခု။ (ဥပမာ - `[1, 2, 3]`)။
*   **2-Dimension (Matrix):** အလျားနဲ့ အနံ (Rows & Columns) ပါတဲ့ ဇယားကွက်။ **အဖြူအမည်းပုံ (Grayscale Image)** တွေဟာ 2D Tensor တွေ ဖြစ်ပါတယ်။ `Shape: (Height, Width)`
*   **3-Dimension (Tensor):** Matrix တွေကို အထပ်လိုက် ထပ်ထားတာပါ။ **ကာလာပုံ (Color Image)** တွေဟာ 3D Tensor တွေ ဖြစ်ပါတယ်။ `Shape: (Channels, Height, Width)`
*   **4-Dimension (Batched Tensor):** 3D Tensor (ပုံတစ်ပုံ) တွေကို အများကြီး ထပ်ပေါင်းထားတာပါ။ AI ကို Train တဲ့အခါ ပုံတစ်ပုံချင်းစီ မသွင်းဘဲ၊ ပုံ ၃၂ ပုံ စသဖြင့် အုပ်စုလိုက် (Batch) သွင်းလေ့ရှိပါတယ်။ အဲ့ဒီအခါ 4D ဖြစ်သွားပါတယ်။ `Shape: (Batch, Channels, Height, Width)`

```python
import torch

# 1D Tensor (Vector)
vector_1d = torch.tensor([1.0, 2.0, 3.0])
print(f"1D Shape: {vector_1d.shape}") # Output: torch.Size([3])

# 2D Tensor (Matrix) - ဥပမာ 2x2 အဖြူအမည်းပုံ
matrix_2d = torch.tensor([[1, 2], [3, 4]])
print(f"2D Shape: {matrix_2d.shape}") # Output: torch.Size([2, 2])
```

---

## 🔄 ၃။ အရေးအကြီးဆုံး အပြောင်းအလဲ (OpenCV vs PyTorch Channels)

ဒီအချက်က Beginner တွေ အများဆုံး Error တက်လေ့ရှိတဲ့ အချက်ပါ။ ပုံတစ်ပုံရဲ့ Channels (အရောင်လွှာများ - RGB) နေရာချထားပုံဟာ OpenCV နဲ့ PyTorch မှာ **လုံးဝ ပြောင်းပြန်** ဖြစ်နေပါတယ်။

*   **OpenCV (Numpy):** `(Height, Width, Channels)` သို့မဟုတ် `(H, W, C)`
    *   *ဥပမာ - `(1080, 1920, 3)` (အရောင်ကို နောက်ဆုံးမှာ ထားသည်)*
*   **PyTorch (Tensor):** `(Channels, Height, Width)` သို့မဟုတ် `(C, H, W)`
    *   *ဥပမာ - `(3, 1080, 1920)` (အရောင်ကို အရှေ့ဆုံးမှာ ထားသည်)*

**ဘာကြောင့် PyTorch က ရှေ့ပို့ထားတာလဲ?**
GPU ပေါ်မှာ Matrix အမြှောက်တွေ လုပ်တဲ့အခါ၊ အရောင်တစ်လွှာချင်းစီကို သီးသန့် တွက်ချက်ရလွယ်အောင် Memory ထဲမှာ Channel ကို ရှေ့ဆုံးကနေ နေရာချထားပေးခြင်း (Memory Optimization) ဖြစ်ပါတယ်။

ဒါကြောင့် OpenCV ကနေ ဖတ်လာတဲ့ ပုံကို PyTorch ထဲ ထည့်တော့မယ်ဆိုရင် Dimension နေရာ ပြန်ရွှေ့ပေးရပါတယ်။ ဒါကို `permute` လို့ ခေါ်ပါတယ်။

```python
import torch
import numpy as np

# ဥပမာ - OpenCV မှ ဖတ်လာသော ပုံ (Numpy Array - HWC)
numpy_image = np.ones((600, 800, 3)) # Height:600, Width:800, Channel:3
print("Numpy Shape (HWC):", numpy_image.shape)

# ၁။ Numpy ကို Tensor ပြောင်းခြင်း
tensor_image = torch.from_numpy(numpy_image)

# ၂။ Dimension နေရာရွှေ့ခြင်း (H, W, C -> C, H, W)
# လက်ရှိ Index: 0=H, 1=W, 2=C
# လိုချင်သော Index: 2=C, 0=H, 1=W
pytorch_tensor = tensor_image.permute(2, 0, 1)

print("PyTorch Shape (CHW):", pytorch_tensor.shape) 
# Output: torch.Size([3, 600, 800])
```

---

## 🗂 ၄။ PyTorch Data Types (dtypes) ကို ကွဲပြားစွာ သိရှိခြင်း

AI တွေကို တွက်ချက်တဲ့အခါ Data Type က အရမ်းအရေးကြီးပါတယ်။ မှားသုံးမိရင် Error ချက်ချင်း တက်ပါတယ်။

1.  **`torch.uint8` (8-bit Unsigned Integer):**
    *   **0 ကနေ 255** အထိ ကိန်းပြည့်တွေသာ ပါဝင်ပါတယ်။
    *   **အသုံးပြုပုံ:** OpenCV ကနေ ဖတ်လာတဲ့ မူရင်းပုံတွေဟာ ဒီအမျိုးအစား ဖြစ်ပါတယ်။ 
    *   *(မှတ်ချက် - ဒီ Type နဲ့ Neural Network ထဲ တိုက်ရိုက် ထည့်သွင်း Train လို့ မရပါဘူး။)*
2.  **`torch.float32` (32-bit Floating Point):**
    *   **ဒသမကိန်းများ** (ဥပမာ - `0.0012`) ပါဝင်ပါတယ်။
    *   **အသုံးပြုပုံ:** Neural Network ရဲ့ Weight တွေ၊ Bias တွေ၊ Loss တွေ အားလုံးကို `float32` နဲ့ တွက်ချက်ပါတယ်။ ဒါကြောင့် မူရင်းပုံ (uint8) တွေကို Model ထဲ မထည့်ခင် `float32` ကို မဖြစ်မနေ ပြောင်းပေးပြီး **0 နှင့် 1 ကြား (Normalization)** ရောက်အောင် 255 နဲ့ စားပေးရပါတယ်။
3.  **`torch.int64` သို့မဟုတ် `torch.long` (64-bit Integer):**
    *   ကိန်းပြည့်အကြီးကြီးတွေပါ။
    *   **အသုံးပြုပုံ:** Model ကို "ဒီပုံဟာ 0 (ခွေး)၊ ဒီပုံဟာ 1 (ကြောင်)" လို့ အဖြေမှန် (Labels) တွေ သတ်မှတ်ပေးတဲ့အခါမှာ အသုံးပြုရပါတယ်။

```python
# မူရင်းပုံ (uint8) ကို Model ထဲထည့်ရန် float32 သို့ ပြောင်းခြင်း
image_tensor = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)

# Float ပြောင်းပြီး 255 ဖြင့် စားခြင်း (0 မှ 1 ကြားသို့ Normalization လုပ်ခြင်း)
normalized_tensor = image_tensor.to(torch.float32) / 255.0

print("New Dtype:", normalized_tensor.dtype) # torch.float32
```

---

## ⚡ ၅။ Tensors များကို GPU ပေါ်သို့ ရွှေ့ခြင်း (Device Move)

PyTorch မှာ `tensor` တွေနဲ့ `model` တွေကို ဆောက်လိုက်ရင် ပုံမှန်အားဖြင့် ကွန်ပျူတာရဲ့ **CPU** ပေါ်မှာပဲ ရှိနေပါတယ်။ GPU အသုံးပြုပြီး မြန်မြန် Train ချင်ရင်တော့ အဲ့ဒီ Data တွေကို GPU (CUDA) ပေါ်ကို `.to()` သုံးပြီး ရွှေ့ပေးရပါတယ်။

```python
# မိမိစက်တွင် GPU ရှိ/မရှိ စစ်ဆေးပြီး သတ်မှတ်ခြင်း
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# CPU ပေါ်ရှိ Tensor တစ်ခု
x_cpu = torch.tensor([1.0, 2.0, 3.0])

# Tensor ကို GPU ပေါ်သို့ ရွှေ့ခြင်း
x_gpu = x_cpu.to(device)

print(x_gpu) # Output: tensor([1., 2., 3.], device='cuda:0')
```
> ⚠️ **သတိပြုရန်:** CPU ပေါ်က Tensor နဲ့ GPU ပေါ်က Tensor ကို ရောပြီး ပေါင်း/နုတ်/မြှောက် လုပ်လို့ **လုံးဝ (လုံးဝ)** မရပါဘူး။ "Expected all tensors to be on the same device" ဆိုတဲ့ Error တက်ပါလိမ့်မယ်။ အမြဲတမ်း နေရာတူအောင် အရင်ရွှေ့ပေးရပါမယ်။

---

## 🪄 ၆။ The Magic: Autograd (Gradient များကို အလိုအလျောက် တွက်ခြင်း)

အရင်အခန်းက Derivative ခက်ခက်ခဲခဲ တွက်ရတဲ့ `dLoss/dw` ကို PyTorch က ဘယ်လောက် လွယ်လွယ်လုပ်ပေးလဲ ကြည့်ရအောင်။

```python
import torch

# x = 2.0, y_true = 10.0, w = 1.0 (Chapter 4.3 က ပုစ္ဆာအတိုင်း)
x = torch.tensor([2.0])
y_true = torch.tensor([10.0])

# Gradient ကို နောက်ပြန်တွက်စေချင်သော Weight တွင် requires_grad=True ဟု ထည့်ပေးရသည်
w = torch.tensor([1.0], requires_grad=True)

# 1. Forward Pass
y_pred = w * x
loss = (y_pred - y_true)**2

# 2. Backward Pass (Chapter 4.3 ကလို Calculus ကိုယ်တိုင်တွက်စရာ မလိုတော့ပါ)
loss.backward()

# Pytorch မှ တွက်ထုတ်ပေးလိုက်သော Gradient (dLoss/dw) ကို ကြည့်ခြင်း
print("Gradient computed by PyTorch:", w.grad) 
# Output: tensor([-32.]) -> (2 * (2-10) * 2 = -32 ကွက်တိ ထွက်လာပါသည်!)
```

---

## 📝 အနှစ်ချုပ် (Key Takeaways)
1. **PyTorch Tensors** များသည် GPU ဖြင့် တွက်ချက်နိုင်ပြီး၊ Gradient များကို အလိုအလျောက် မှတ်သားတွက်ချက်ပေးနိုင်သော (Autograd) Matrix များ ဖြစ်သည်။
2. AI ကို Train သောအခါ Tensors များ၏ Dimension ပုံစံမှာ **BCHW (Batch, Channels, Height, Width)** 4D Format ဖြစ်သည်။
3. OpenCV ၏ Channel `(H, W, C)` အား PyTorch ၏ `(C, H, W)` သို့ `permute(2, 0, 1)` ဖြင့် ပြောင်းပေးရမည်။
4. တွက်ချက်မှုများ ပြုလုပ်ရန် Images များကို `uint8` မှ `float32` သို့ ပြောင်း၍ Normalization (255 ဖြင့်စားခြင်း) ပြုလုပ်ရမည်။
5. Tensors များကို ပေါင်း/မြှောက်ရာတွင် အားလုံးသည် **Device တစ်ခုတည်း (CPU သို့မဟုတ် GPU တစ်ခုတည်း)** ပေါ်တွင် ရှိနေရမည်။

---

# 🎉 Module 4 ပြီးဆုံးပါပြီ!
ဂုဏ်ယူပါတယ်။ AI တွေရဲ့ နောက်ကွယ်က သင်္ချာသဘောတရားတွေ (Neural Networks, Forward Pass, Loss, Backpropagation) နဲ့ PyTorch ရဲ့ အခြေခံအုတ်မြစ်တွေကို အပြည့်အဝ နားလည်သွားပါပြီ။

> **🔜 Next Step (Enter the CNN Era):**
> ရှေ့ဆက်မယ့် **Part 2: Module 5 (Convolutional Neural Networks)** အပိုင်းမှာတော့ "သာမန် Artificial Neural Network တွေက ဘာလို့ ပုံတွေ (Images) ကို ခွဲခြားဖို့ အဆင်မပြေတာလဲ?" ဆိုတာကို လေ့လာပြီး၊ Computer Vision ရဲ့ အကြီးမားဆုံး တော်လှန်ရေးဖြစ်တဲ့ **CNN (ConvNets) Architectures** တွေဆီကို စတင် ချီတက်သွားကြပါမယ်။

---
*If you find this useful, don't forget to ⭐ star the repository!*

---

