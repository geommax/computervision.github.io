


GitHub Knowledge Sharing Page အတွက် **Part 5** ရဲ့ ပထမဆုံးအခန်းဖြစ်တဲ့ **Module 10: Chapter 10.1** ကို ရေးသားပေးလိုက်ပါတယ်။ 

ဒီအပိုင်းကစပြီး စာတမ်းတွေ၊ စမ်းသပ်ခန်းတွေထဲက AI Model ကြီးကို လက်တွေ့ကမ္ဘာက (User တွေ အသုံးပြုမယ့်) Server တွေ၊ Device တွေပေါ်မှာ အမြန်ဆုံး အလုပ်လုပ်နိုင်အောင် ဘယ်လို ပြောင်းလဲပြင်ဆင်ရမလဲ ဆိုတဲ့ **Software Engineering & Deployment** သဘောတရားတွေကို လေ့လာသွားကြပါမယ်။

---

# 🚀 PART 5: Production, MLOps & Deployment (လက်တွေ့နယ်ပယ်သို့ ရောက်ရှိခြင်း)

# 📘 Chapter 10.1: Model Optimization & Exporting (ONNX နှင့် TensorRT အသုံးပြုခြင်း)

Python တွေ၊ PyTorch တွေသုံးပြီး AI Model ကြီးတစ်ခုကို Train လို့ ပြီးသွားပါပြီ။ ရလာတဲ့ `.pt` သို့မဟုတ် `.pth` ဖိုင်ကြီးကိုယူပြီး ကုမ္ပဏီရဲ့ Web Server ပေါ်ကို တိုက်ရိုက်တင်ပြီး သုံးလိုက်လို့ ရမလား?

**အဖြေက "မသုံးသင့်ပါဘူး"။**

ဘာကြောင့်လဲဆိုတော့ PyTorch ဟာ "လေ့ကျင့်သင်ယူဖို့ (Training)" အတွက် အထူးဖန်တီးထားတာ ဖြစ်လို့ပါ။ သူ့ရဲ့ ဖိုင်တွေထဲမှာ Backpropagation တွက်ဖို့ Gradient အချက်အလက်တွေ၊ Optimizer တွေ အများကြီး ရောပါနေပါတယ်။ ဒါတွေဟာ ပုံတစ်ပုံကို "ခန့်မှန်းရုံသက်သက် (Inference)" လုပ်မယ့် Production ပတ်ဝန်းကျင်မှာ လုံးဝ မလိုအပ်တဲ့အပြင်၊ Model ကို လေးလံနှေးကွေးစေပါတယ်။ 

ဒါကြောင့် Model ကို အမြန်ဆုံး အလုပ်လုပ်နိုင်အောင် **Optimization (မြန်နှုန်းမြှင့်တင်ခြင်း)** အရင်လုပ်ပေးရပါတယ်။ 

---

## 📄 ၁။ ONNX ဆိုတာ ဘာလဲ? (The "PDF" of Neural Networks)

**ONNX (Open Neural Network Exchange)** ဆိုတာ AI Model တွေအတွက် Standard Format (စံသတ်မှတ်ချက်) တစ်ခု ဖြစ်ပါတယ်။ 

**ဥပမာ -** 
သင်က စာစီစာကုံး တစ်ပုဒ်ကို Microsoft Word (`.docx`) နဲ့ ရေးတယ်။ တခြားလူက ကြည့်ချင်တဲ့အခါ သူ့စက်ထဲမှာ Word မရှိရင် ဖွင့်လို့မရပါဘူး။ ဒါပေမယ့် အဲ့ဒီဖိုင်ကို PDF (`.pdf`) ပြောင်းပြီး ပို့လိုက်ရင် ဘယ်ဖုန်း၊ ဘယ်ကွန်ပျူတာကမဆို အလွယ်တကူ၊ မြန်မြန်ဆန်ဆန် ဖွင့်ကြည့်လို့ ရသွားပါတယ်။

AI လောကမှာလည်း အတူတူပါပဲ။
*   **PyTorch / TensorFlow:** Microsoft Word နဲ့ တူပါတယ်။ ပြင်ရလွယ်တယ်၊ Train ရလွယ်တယ်။ ဒါပေမယ့် လေးလံတယ်။
*   **ONNX:** PDF နဲ့ တူပါတယ်။ Weight တွေကို Update ပြန်လုပ်လို့ (Train လို့) မရတော့ပါဘူး။ ဒါပေမယ့် အလွန်ပေါ့ပါးသွားပြီး C++, C#, Java, JavaScript ကြိုက်တဲ့ Programming Language ကနေ PyTorch ကြီး သွင်းစရာမလိုဘဲ လှမ်းခေါ်သုံးလို့ ရသွားပါပြီ။

---

## 🏎️ ၂။ TensorRT ဆိုတာ ဘာလဲ? (The F1 Engine for AI)

ONNX က Format ပြောင်းပေးတာဆိုရင်၊ **TensorRT** ကတော့ NVIDIA ကုမ္ပဏီကနေ သူတို့ရဲ့ GPU တွေပေါ်မှာ AI တွေကို အမြန်ဆုံး ပြေးနိုင်အောင် တည်ဆောက်ပေးထားတဲ့ **ပြိုင်ကားအင်ဂျင် (Inference Engine)** ကြီး ဖြစ်ပါတယ်။

TensorRT က Model ကို အမြန်ဆုံးဖြစ်အောင် အဓိက နည်းလမ်း ၂ ခုနဲ့ ပြုပြင်ပေးပါတယ်။

1.  **Layer Fusion (အလွှာများကို ပေါင်းစပ်ခြင်း):**
    Convolution -> BatchNorm -> ReLU ဆိုတဲ့ အလွှာ ၃ ခုကို PyTorch က အဆင့် ၃ ဆင့် ခွဲတွက်ပါတယ်။ TensorRT က အဲ့ဒီ ၃ ခုကို သင်္ချာနည်းအရ ဖိသိပ်ပြီး (Fuse လုပ်ပြီး) အဆင့် ၁ ဆင့်တည်း အဖြစ် ပြောင်းပစ်လိုက်ပါတယ်။ အချိန်အများကြီး သက်သာသွားပါတယ်။
2.  **Precision Calibration (Quantization - ဂဏန်းအရွယ်အစား လျှော့ချခြင်း):**
    မူလ AI Model တွေဟာ ဂဏန်းတွေကို `Float32` (ဒသမ ၃၂-ဘစ်) နဲ့ တွက်ပါတယ်။ အရမ်းတိကျပေမယ့် အလွန်လေးပါတယ်။ TensorRT က အဲ့ဒီဂဏန်းတွေကို Accuracy သိပ်မကျစေဘဲ `Float16` ဒါမှမဟုတ် `INT8` (ကိန်းပြည့် ၈-ဘစ်) အထိ အလိုအလျောက် ချုံ့ပစ်လိုက်ပါတယ်။ ဒါကြောင့် **မြန်နှုန်း (FPS) ဟာ မူလထက် ၃ ဆ ကနေ ၅ ဆ အထိ မြန်ဆန်သွားပါတယ်။**

![PyTorch to ONNX to TensorRT](https://via.placeholder.com/800x250?text=PyTorch+%28Training%29+-%3E+ONNX+%28Standard+Format%29+-%3E+TensorRT+%28Max+Speed+Inference%29)

---

## 💻 ၃။ PyTorch Model ကို ONNX သို့ ပြောင်းလဲခြင်း (Code Example)

PyTorch Model ကို ONNX ပြောင်းဖို့ အလွန်လွယ်ကူပါတယ်။ ဒါပေမယ့် မပြောင်းခင်မှာ အရေးအကြီးဆုံး အချက်က **Model ကို တွက်ချက်မှု ရပ်တန့်ရန် `model.eval()`** မဖြစ်မနေ လုပ်ပေးဖို့ လိုအပ်ပါတယ်။

```python
import torch
import torchvision.models as models

def export_to_onnx():
    # ၁။ Train လုပ်ပြီးသား PyTorch Model ကို ခေါ်ယူခြင်း (ဥပမာ - ResNet18)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # ⚠️ အရေးကြီးဆုံးအဆင့်: Training mode မှ Evaluation (Inference) mode သို့ ပြောင်းခြင်း
    # ဤသို့မလုပ်ပါက Dropout နှင့် BatchNorm များသည် Training အတိုင်း အလုပ်လုပ်နေမည်ဖြစ်သည်။
    model.eval()

    # ၂။ Model ထဲသို့ ဝင်မည့် ပုံအရွယ်အစား အတိုင်း (Dummy Input) တစ်ခု ဖန်တီးခြင်း
    # ဥပမာ -[Batch_size=1, Channels=3, Height=224, Width=224]
    dummy_input = torch.randn(1, 3, 224, 224)

    # ၃။ ONNX format သို့ ပြောင်းလဲ၍ Export လုပ်ခြင်း
    onnx_file_path = "resnet18_optimized.onnx"
    
    torch.onnx.export(
        model,                  # Export လုပ်မည့် Model
        dummy_input,            # Dummy input ပုံစံ
        onnx_file_path,         # ထွက်လာမည့် ဖိုင်အမည်
        export_params=True,     # Weights များကိုပါ တစ်ပါတည်း ထည့်သိမ်းမည်
        opset_version=11,       # ONNX version
        input_names=['input'],  # Input အမည် (နောက်ပိုင်း C++ စသည်ဖြင့် ခေါ်သုံးရန်)
        output_names=['output'],# Output အမည်
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} 
        # Batch size ကို အသေမထားဘဲ လိုသလို ပြောင်းနိုင်ရန် Dynamic အဖြစ် ထားခြင်း
    )

    print(f"🎉 Model successfully exported to {onnx_file_path}")

if __name__ == "__main__":
    export_to_onnx()
```

*(ဒီ Code ကို Run ပြီးသွားရင် ရလာတဲ့ `resnet18_optimized.onnx` ဖိုင်လေးကို Backend (Node.js, C#, Python ONNXRuntime) တွေကနေ ပေါ့ပေါ့ပါးပါးနဲ့ အမြန်ဆုံး ယူသုံးလို့ ရသွားပါပြီ။)*

---

## ⏱️ ၄။ YOLOv8 ကို ONNX / TensorRT ပြောင်းခြင်း

Ultralytics (YOLO) မှာဆိုရင်တော့ ပိုလို့တောင် လွယ်ကူပါသေးတယ်။ Command တစ်ကြောင်းတည်းနဲ့ Export လုပ်နိုင်ပါတယ်။

```bash
# ONNX သို့ ပြောင်းရန်
yolo export model=yolov8n.pt format=onnx

# TensorRT (Engine) သို့ ပြောင်းရန် (NVIDIA GPU ရှိရန် လိုအပ်သည်)
yolo export model=yolov8n.pt format=engine half=True
# half=True ဆိုသည်မှာ Float32 မှ Float16 သို့ ချုံ့၍ မြန်နှုန်းမြှင့်မည်ဟု ဆိုလိုသည်
```

---

## 📝 အနှစ်ချုပ် (Key Takeaways)
1.  **Optimization** သည် AI Model များကို Production (လုပ်ငန်းခွင်) တွင် အမြန်ဆုံးနှင့် အပေါ့ပါးဆုံး ဖြစ်အောင် ပြုပြင်ပေးသော မရှိမဖြစ် အဆင့်ဖြစ်သည်။
2.  **ONNX** သည် Framework ပေါင်းစုံ (PyTorch, TF) နှင့် Programming Language ပေါင်းစုံ အကြား အလွယ်တကူ ဖလှယ်အသုံးပြုနိုင်သော Universal AI "PDF" Format ဖြစ်သည်။
3.  **TensorRT** သည် NVIDIA GPU များပေါ်တွင် Layer များ ပေါင်းစပ်ခြင်းနှင့် ဂဏန်းအရွယ်အစား ချုံ့ခြင်း (Quantization) တို့ကို ပြုလုပ်ပေးပြီး Inference Speed ကို အမြင့်ဆုံးအထိ မြှင့်တင်ပေးသည်။
4.  PyTorch မှ Export မလုပ်မီ **`model.eval()`** သို့ ပြောင်းရန် လုံးဝ (လုံးဝ) မမေ့သင့်ပါ။

> **🔜 Next Step:**
> "အိုကေ.. Model လည်း ပေါ့ပါးသွားပြီဆိုတော့ ဒီ Model ကြီးကို Server အကြီးကြီးတွေမှာ မဟုတ်ဘဲ၊ ကင်မရာလေးတွေမှာ တပ်ထားတဲ့ စက်အသေးလေးတွေပေါ်ကို ဘယ်လို တင်မလဲ?" 
> ရှေ့ဆက်မယ့် **Chapter 10.2: Deploying on Edge Devices (Raspberry Pi & Jetson Nano)** မှာ Hardware အသေးလေးတွေပေါ်မှာ Computer Vision သုံးတဲ့ သဘောတရားတွေကို ဆက်လက် လေ့လာသွားကြပါမယ်။

---
*If you find this useful, don't forget to ⭐ star the repository!*

---
