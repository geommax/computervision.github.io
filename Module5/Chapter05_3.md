


GitHub Knowledge Sharing Page အတွက် **Module 5: Chapter 5.3** ကို ဆက်လက် ရေးသားပေးလိုက်ပါတယ်။ ဒီအခန်းဟာ လက်တွေ့ လုပ်ငန်းခွင် (Industry) ထဲကို ရောက်တဲ့အခါ AI Engineer တိုင်း နေ့စဉ်နဲ့အမျှ အသုံးပြုရမယ့် အလွန်အရေးပါတဲ့ နည်းပညာဖြစ်တဲ့ **Transfer Learning** အကြောင်းကို လေ့လာသွားမယ့် အခန်းဖြစ်ပါတယ်။

---

# 📘 Chapter 5.3: Modern CNN Architectures & Transfer Learning (ခေတ်သစ် CNN မော်ဒယ်များနှင့် လွှဲပြောင်းသင်ယူခြင်း)

အရင်အခန်းက `SimpleCNN` လေးတစ်ခုကို ကိုယ်တိုင် အစအဆုံး (From Scratch) ရေးသားခဲ့ပါတယ်။ ဒါပေမယ့် လက်တွေ့ကမ္ဘာက ရှုပ်ထွေးတဲ့ ပြဿနာတွေ (ဥပမာ - ကင်ဆာဆဲလ် ရှာဖွေခြင်း၊ ကားအမျိုးအစား ခွဲခြားခြင်း) တွေကို ဖြေရှင်းဖို့အတွက် အလွှာ ၃/၄ လွှာလောက်နဲ့ မလုံလောက်တော့ပါဘူး။ 

"ဒါဆိုရင် အလွှာ ၁၀၀ လောက်ကို ကိုယ်တိုင် ထိုင်ရေးပြီး Train ရမှာလား?"
ကံကောင်းစွာနဲ့ပဲ အဲ့ဒီလို လုပ်စရာမလိုပါဘူး။ Google, Microsoft, Meta စတဲ့ ကုမ္ပဏီကြီးတွေက ကမ္ဘာ့အတော်ဆုံး သုတေသီတွေ ပေါင်းပြီး Supercomputers ကြီးတွေနဲ့ လနဲ့ချီ Train ထားတဲ့ အလွန်တော်တဲ့ Model ကြီးတွေ ရှိပါတယ်။ အဲ့ဒီ Model ကြီးတွေကို အလွယ်တကူ ယူသုံးတာကို **Transfer Learning** လို့ ခေါ်ပါတယ်။ 

မသုံးခင်မှာ ဘယ် Model က ဘာအတွက်ကောင်းလဲဆိုတာ အရင် လေ့လာကြည့်ရအောင်။

---

## 🏛️ ၁။ CNN သမိုင်းကြောင်းနှင့် နာမည်ကြီး Model များ (Evolution of CNNs)

Computer Vision လောကကို တော်လှန်ခဲ့တဲ့ အထင်ကရ Model တွေကတော့ - 

1. **VGG (2014):** အလွှာ (Layers) ၁၆ ခု ကနေ ၁၉ ခု အထိ ရှိပါတယ်။ `3x3` Convolution လေးတွေကိုပဲ ထပ်ခါထပ်ခါ သုံးသွားတဲ့ အလွန်ရှင်းလင်းတဲ့ ပုံစံပါ။ ဒါပေမယ့် Parameters (Weight) သန်း ၁၃၀ ကျော် ပါတဲ့အတွက် အလွန်လေးလံပြီး နှေးကွေးပါတယ်။
2. **ResNet (Residual Networks - 2015):** Microsoft က တီထွင်ခဲ့တာဖြစ်ပြီး Computer Vision လောကရဲ့ **ဂိမ်းချိန်းဂျာ (Game Changer)** ပါပဲ။ အရင်က အလွှာတွေ အရမ်းများလာရင် Gradient တွေ ပျောက်သွားပြီး (Vanishing Gradient) Model က ဆက်မသင်ယူနိုင်တော့တဲ့ ပြဿနာ ရှိပါတယ်။ ResNet က အလွှာတစ်ခုကို ကျော်ပြီး နောက်တစ်ခုဆီ တိုက်ရိုက်ဆက်သွယ်ပေးတဲ့ **Skip Connections (Shortcut)** နည်းပညာကို တီထွင်ခဲ့ပါတယ်။ ဒါကြောင့် အလွှာ ၅၀၊ ၁၀၁ ကနေ ၁၅၂ ခုအထိ အလွန်နက်ရှိုင်းအောင် Train လို့ ရသွားပါတယ်။
3. **MobileNet (2017) / EfficientNet (2019):** ResNet တွေက တော်ပေမယ့် ဖုန်းတွေ၊ Raspberry Pi တွေပေါ်မှာ Run ဖို့ ကြီးမားလွန်းပါတယ်။ ဒါကြောင့် Google ကနေ Parameters အနည်းငယ်နဲ့ အဖြေမှန်ကန်မှု (Accuracy) မြင့်မားတဲ့ ဒီ Model တွေကို ထပ်မံတီထွင်ခဲ့ပါတယ်။ (Edge AI အတွက် အများဆုံး သုံးပါတယ်)။

---

## 🧠 ၂။ Transfer Learning ဆိုတာ ဘာလဲ? (လွှဲပြောင်းသင်ယူခြင်း)

**ဥပမာ -** စက်ဘီးစီးတတ်တဲ့ လူတစ်ယောက်ဟာ ဆိုင်ကယ် စီးတတ်ဖို့အတွက် အစကနေ ပြန်သင်စရာ မလိုပါဘူး။ Balance (ဟန်ချက်) ထိန်းတတ်တဲ့ အခြေခံစွမ်းရည် (Knowledge) ကို လွှဲပြောင်း (Transfer) ယူလိုက်ပြီး၊ လီဗာဆွဲတာလောက်ကိုပဲ အနည်းငယ် ထပ်သင် (Fine-tune) လိုက်ရုံပါပဲ။

Deep Learning မှာလည်း ဒီအတိုင်းပါပဲ။ 
ကုမ္ပဏီကြီးတွေဟာ **ImageNet** လို့ခေါ်တဲ့ (ပုံပေါင်း ၁.၂ သန်းကျော်၊ ခွေး၊ ကြောင်၊ ကား၊ လေယာဉ် စသည့် အရာဝတ္ထုပေါင်း ၁၀၀၀ ပါဝင်သော) Dataset ကြီးနဲ့ Model ကြီးတွေကို လနဲ့ချီ Train ထားပါတယ်။ ဒါကြောင့် အဲ့ဒီ Model ကြီးတွေရဲ့ Conv Layers (Feature Extractors) တွေဟာ ပုံတွေထဲက အနားသတ်တွေ၊ ပုံသဏ္ဌာန်တွေကို ရှာဖွေရာမှာ **ပါရဂူမြောက် ကျွမ်းကျင်နေပါပြီ။** 

ကျွန်တော်တို့က ကိုယ်ပိုင် "မြွေ အဆိပ်ရှိ/မရှိ ခွဲခြားတဲ့ AI" လုပ်ချင်တယ်ဆိုပါစို့။ မြွေပုံတွေကို အစကနေ သွင်းပြီး Conv Layer တွေကို ပြန် Train နေမယ့်အစား၊ ImageNet မှာ Train ထားတဲ့ အဲ့ဒီ Model ကြီးကို ယူလိုက်ပါတယ်။ ပြီးရင် နောက်ဆုံးက **"ဆုံးဖြတ်ချက်ချတဲ့ Output Layer လေး တစ်ခုကိုပဲ ဖြုတ်ပြီး၊ မိမိလိုချင်တဲ့ Class (၂ မျိုး) ပြောင်းကာ အနည်းငယ် ပြန်လေ့ကျင့်ပေးလိုက်ခြင်း"** ကို Transfer Learning လို့ ခေါ်ပါတယ်။

---

## 🛠️ ၃။ Transfer Learning ၏ အဆင့်များ (Freezing & Fine-Tuning)

Transfer Learning လုပ်ရာမှာ အဓိက အဆင့် ၃ ဆင့် ရှိပါတယ်။

1. **Load Pre-trained Model:** အသင့် Train ထားသော Model (ဥပမာ - ResNet18) ၏ Architecture နှင့် Weights များကို Download ဆွဲယူခြင်း။
2. **Freeze the Conv Layers:** ရှေ့ပိုင်းမှာရှိတဲ့ Feature ရှာတဲ့ Conv Layer တွေက အရမ်းတော်နေပြီ ဖြစ်တဲ့အတွက် သူတို့ရဲ့ Weight တွေကို ဆက်မပြောင်းလဲစေဖို့ "Freeze (ခဲထားခြင်း/သော့ခတ်ခြင်း)" ပြုလုပ်ရပါတယ်။
3. **Replace the FC Layer (Head):** နောက်ဆုံး `nn.Linear` အလွှာက မူလက Class ၁၀၀၀ ခွဲဖို့ ဖြစ်နေပါတယ်။ အဲ့ဒါကို ဖြုတ်ပြီး၊ မိမိခွဲခြားလိုသော အရေအတွက် (ဥပမာ ၂ မျိုး - Poisonous vs Non-Poisonous) ဖြင့် အစားထိုးကာ အဆိုပါ နောက်ဆုံးအလွှာလေးကိုသာ Train လုပ်ခြင်း ဖြစ်ပါတယ်။

---

## 💻 ၄။ PyTorch ဖြင့် Transfer Learning လက်တွေ့ရေးသားခြင်း

PyTorch တွင် နာမည်ကြီး Model များကို `torchvision.models` ထဲမှ အလွယ်တကူ ခေါ်ယူအသုံးပြုနိုင်ပါတယ်။ 

*(အောက်ပါ Code သည် ResNet18 ကို အသုံးပြု၍ ခွေးနှင့်ကြောင် (Class ၂ မျိုးတည်း) ခွဲမည့် Model အဖြစ် ပြောင်းလဲထားသော Code ဖြစ်သည်။)*

```python
import torch
import torch.nn as nn
from torchvision import models

def create_transfer_learning_model():
    # ၁။ Pre-trained Model (ResNet18) ကို ခေါ်ယူခြင်း
    # weights='IMAGENET1K_V1' ဟုထည့်မှသာ ImageNet ဖြင့် Train ထားသော Weights များ ပါလာမည်
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    print("Original FC Layer:", model.fc)
    # တွေ့ရမည့် Output: Linear(in_features=512, out_features=1000, ...) -> Class 1000 မျိုး

    # ၂။ Conv Layers များကို Freeze လုပ်ခြင်း (Weight များ မပြောင်းလဲစေရန်)
    for param in model.parameters():
        param.requires_grad = False  # Gradient တွက်ချက်ခြင်းကို ပိတ်ထားလိုက်သည်
        
    # ၃။ နောက်ဆုံး Fully Connected (FC) Layer ကို မိမိလိုချင်သလို ပြောင်းလဲခြင်း
    # ResNet ရဲ့ နောက်ဆုံး layer ကို 'fc' ဟု နာမည်ပေးထားသည် (VGG တွင် 'classifier' ဟုခေါ်သည်)
    num_features = model.fc.in_features # ဝင်လာမည့် feature အရေအတွက် (512)
    
    # Class ၂ မျိုးတည်း ခွဲမည့် Linear Layer အသစ်ဖြင့် အစားထိုးခြင်း 
    # (အစားထိုးလိုက်သော အလွှာသစ်သည် requires_grad=True အလိုအလျောက် ဖြစ်နေမည်)
    model.fc = nn.Linear(num_features, 2) 
    
    print("\nModified FC Layer:", model.fc)
    # တွေ့ရမည့် Output: Linear(in_features=512, out_features=2, ...) -> Class 2 မျိုး

    return model

# Model ကို တည်ဆောက်စမ်းသပ်ခြင်း
my_custom_model = create_transfer_learning_model()

# ဥပမာ - ပုံတစ်ပုံ ဝင်လာသည်ဟု သဘောထားပါ[Batch=1, Channels=3, H=224, W=224]
dummy_input = torch.randn(1, 3, 224, 224)
output = my_custom_model(dummy_input)

print(f"\nFinal Output Shape: {output.shape}") 
# Output: torch.Size([1, 2]) -> အောင်မြင်စွာ Class 2 မျိုး ထွက်လာပါပြီ!
```

**💡 ဒီလိုလုပ်ခြင်းရဲ့ အကျိုးကျေးဇူး:**
ပုံတွေ သိန်းနဲ့ချီ မလိုတော့ပါဘူး။ ကိုယ်ခွဲချင်တဲ့ ပုံ ၅၀၀ လောက်နဲ့တင် Training အချိန် ၅ မိနစ်၊ ၁၀ မိနစ်လောက် ပေးလိုက်ရုံနဲ့ အလွန်တိကျတဲ့ AI Model တစ်ခုကို တည်ဆောက်နိုင်သွားပါပြီ။

---

## 📝 အနှစ်ချုပ် (Key Takeaways)
1. **ResNet** သည် Vanishing Gradient ပြဿနာကို Skip Connections ဖြင့် ဖြေရှင်းထားပြီး လက်ရှိအချိန်ထိ အသုံးအများဆုံး စံပြ Model တစ်ခုဖြစ်သည်။
2. **Transfer Learning** သည် ImageNet ကဲ့သို့သော Dataset ကြီးများဖြင့် လေ့ကျင့်ထားသော Model များကို ယူ၍ မိမိ၏ သီးသန့် Project များအတွက် ပြန်လည်အသုံးပြုခြင်း ဖြစ်သည်။
3. အစောပိုင်း အလွှာများ (Feature Extractors) ကို **Freeze (`requires_grad = False`)** ပြုလုပ်၍ နောက်ဆုံး Output Layer (Head) ကို မိမိလိုချင်သော Class အရေအတွက်ဖြင့် အစားထိုးရသည်။
4. Transfer Learning ကို အသုံးပြုခြင်းဖြင့် Data နည်းနည်း၊ အချိန်တိုတိုဖြင့် Accuracy အလွန်ကောင်းသော Model များကို တည်ဆောက်နိုင်သည်။

> **🔜 Next Step:**
> သီအိုရီတွေ၊ Architecture တွေ အားလုံး သိသွားပြီဆိုတော့... ဒီ Model ကြီးတွေထဲကို ကိုယ်ပိုင် Data တွေ (ဥပမာ- Folder ထဲမှာ သိမ်းထားတဲ့ ပုံတွေ) ဘယ်လို သွင်းမလဲ? PyTorch ရဲ့ DataLoader ဆိုတာ ဘာလဲ? Loop တွေ ပတ်ပြီး တကယ် Model ကြီးကို ဘယ်လို Train မလဲ ဆိုတာကို **Chapter 5.4: PyTorch Image Classification Pipeline (Project)** မှာ End-to-End လက်တွေ့ ရေးသားသွားကြပါမယ်။

---
*If you find this useful, don't forget to ⭐ star the repository!*

---
