


GitHub Knowledge Sharing Page အတွက် **Module 5 ၏ နောက်ဆုံးအခန်းဖြစ်သော Chapter 5.4** ကို ဆက်လက် ရေးသားပေးလိုက်ပါတယ်။ ဒီအခန်းဟာ အရင်အခန်းတွေမှာ သင်ခဲ့တဲ့ သီအိုရီတွေအားလုံးကို ပေါင်းစပ်ပြီး၊ လက်တွေ့ လုပ်ငန်းခွင်မှာ အသုံးပြုတဲ့ **End-to-End PyTorch Training Pipeline** တစ်ခုကို အစအဆုံး တည်ဆောက်ပြသွားမယ့် အလွန်အရေးပါတဲ့ Project အခန်း ဖြစ်ပါတယ်။

---

# 🚀 PART 2: Module 5 - Convolutional Neural Networks (CNN) & Classification

# 📘 Chapter 5.4: PyTorch Image Classification Pipeline (Project အစအဆုံး တည်ဆောက်ခြင်း)

Computer Vision Project တစ်ခုကို လက်တွေ့ရေးသားတော့မယ်ဆိုရင် အဆင့် ၄ ဆင့် (4 Core Steps) ကို အမြဲတမ်း ဖြတ်သန်းရပါတယ်။ ဒီအခန်းမှာ ခွေးနဲ့ကြောင် ခွဲခြားမယ့် (Cats vs Dogs) AI Model တစ်ခုကို Transfer Learning အသုံးပြုပြီး စနစ်တကျ တည်ဆောက်သွားကြပါမယ်။

---

## 📂 ၁။ Dataset ဖွဲ့စည်းပုံနှင့် Data Augmentation (အချက်အလက်များ ပြင်ဆင်ခြင်း)

PyTorch ကို အသုံးပြုမယ်ဆိုရင် ပုံတွေကို Folder တွေနဲ့ စနစ်တကျ ခွဲခြားထားဖို့ လိုအပ်ပါတယ်။ အောက်ပါအတိုင်း Training (သင်ယူရန်) နှင့် Validation (စစ်ဆေးရန်) Folder များ ခွဲထားရပါမယ်။

```text
dataset/
├── train/
│   ├── cats/ (ကြောင်ပုံများ...)
│   └── dogs/ (ခွေးပုံများ...)
└── val/
    ├── cats/ (ကြောင်ပုံများ...)
    └── dogs/ (ခွေးပုံများ...)
```

**Data Augmentation & Transforms:**
AI ဟာ ပုံတွေကို အလွတ်ကျက်သွားတာမျိုး (Overfitting) မဖြစ်စေဖို့အတွက်၊ ပုံတွေကို လှည့်တာ၊ ပြောင်းပြန်လှန်တာ စတဲ့ **ပုံပွားခြင်း (Data Augmentation)** တွေကို လုပ်ပေးရပါတယ်။ ပြီးရင် Model နားလည်တဲ့ `Tensor` အဖြစ်ပြောင်းပြီး Normalization လုပ်ရပါတယ်။ ဒါတွေကို `torchvision.transforms` ကနေ အလွယ်တကူ လုပ်ဆောင်နိုင်ပါတယ်။

```python
from torchvision import transforms

# Training အတွက် (Data Augmentation ပါဝင်သည်)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # CNN များသည် အရွယ်အစားတူညီရန် လိုအပ်သည်
    transforms.RandomHorizontalFlip(p=0.5), # 50% သောပုံများကို ဘယ်/ညာ လှန်မည်
    transforms.RandomRotation(10), # ၁၀ ဒီဂရီ စောင်းမည်
    transforms.ToTensor(), # (H,W,C) မှ (C,H,W) Tensor သို့ပြောင်းမည်၊ 0-1 ကြားထားမည်
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet စံနှုန်းဖြင့် Normalization လုပ်ခြင်း
])

# Validation အတွက် (စစ်ဆေးရုံသာဖြစ်သဖြင့် Augmentation မလိုပါ)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 🚚 ၂။ Dataset နှင့် DataLoader (Data များကို Model ထံသို့ ပို့ဆောင်ခြင်း)

Folder ထဲက ပုံတွေကို ဖတ်ဖို့ `ImageFolder` ကိုသုံးပါတယ်။ ဖတ်လာတဲ့ ပုံတွေကို GPU ပေါ်ကို ၃၂ ပုံ တစ်သုတ် စသဖြင့် အုပ်စုလိုက် (Batch) ပို့ဆောင်ပေးဖို့ `DataLoader` ဆိုတဲ့ ကားလေးတွေကို အသုံးပြုရပါတယ်။

```python
from torchvision import datasets
from torch.utils.data import DataLoader

# Folder ထဲမှ ပုံများကို ဖတ်ခြင်း
train_dataset = datasets.ImageFolder(root='dataset/train', transform=train_transforms)
val_dataset = datasets.ImageFolder(root='dataset/val', transform=val_transforms)

print("Classes:", train_dataset.classes) # Output: ['cats', 'dogs']

# Model ထံသို့ 32 ပုံ တစ်သုတ်စီ ပို့ဆောင်ပေးမည့် DataLoader များ တည်ဆောက်ခြင်း
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Train တွင် မွှေနှောက် (Shuffle) ရန်လိုသည်
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

---

## ⚙️ ၃။ Model, Loss နှင့် Optimizer ပြင်ဆင်ခြင်း

Chapter 5.3 က သင်ခဲ့တဲ့ Transfer Learning (ResNet18) နဲ့ Chapter 4.3 က Optimizer (Adam) တွေကို အခု ပေါင်းစပ်ပါတော့မယ်။

```python
import torch
import torch.nn as nn
from torchvision import models

# CPU သို့မဟုတ် GPU ရွေးချယ်ခြင်း
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transfer Learning Model တည်ဆောက်ခြင်း
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2) # Cats နှင့် Dogs (2 မျိုး)
model = model.to(device) # Model ကို GPU ပေါ်တင်ခြင်း

# Loss Function နှင့် Optimizer
criterion = nn.CrossEntropyLoss() # Classification အတွက်
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 🔁 ၄။ Training & Validation Loop (အဓိက သင်ယူခြင်း လုပ်ငန်းစဉ်)

AI Model တစ်ခုလုံးရဲ့ အသက်သွေးကြောဖြစ်တဲ့ Loop ကြီး ဖြစ်ပါတယ်။ Data တွေကို Batch လိုက်ယူမယ်၊ ခန့်မှန်းမယ်၊ အမှားတွက်မယ်၊ Weights တွေ ပြင်မယ် ဆိုတဲ့ လုပ်ငန်းစဉ် (Forward -> Loss -> Backward -> Step) ကို အကြိမ်ကြိမ် လုပ်ဆောင်မှာ ဖြစ်ပါတယ်။

```python
num_epochs = 5 # Data အားလုံးကို ၅ ခေါက် အပြည့် လေ့ကျင့်မည်

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 10)
    
    # ==========================================
    # 1. Training Phase
    # ==========================================
    model.train() # Model ကို Training လုပ်မည်ဟု အသိပေးခြင်း
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for images, labels in train_loader:
        # Data များကို GPU ပေါ်တင်ခြင်း
        images, labels = images.to(device), labels.to(device)
        
        # Optimizer အဟောင်းများကို သုည (0) ပြန်လုပ်ခြင်း (မရှိမဖြစ်လိုအပ်သည်)
        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward Pass (Gradient များ တွက်ချက်ခြင်း)
        loss.backward()
        
        # Optimizer Step (Weights များကို Update လုပ်ခြင်း)
        optimizer.step()
        
        # မှတ်တမ်းတင်ခြင်း
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1) # အများဆုံး ရမှတ်ရှိသော Class ကို ရွေးခြင်း
        correct_preds += torch.sum(preds == labels.data)
        total_preds += labels.size(0)
        
    train_loss = running_loss / total_preds
    train_acc = correct_preds.double() / total_preds
    print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

    # ==========================================
    # 2. Validation Phase
    # ==========================================
    model.eval() # Model ကို စစ်ဆေးသည့် အဆင့်သို့ ပြောင်းခြင်း (Weight Update မလုပ်တော့ပါ)
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad(): # Validation တွင် Gradient တွက်စရာမလိုသဖြင့် ပိတ်ထားခြင်း (Memory သက်သာစေသည်)
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)
            val_total += labels.size(0)
            
    epoch_val_loss = val_loss / val_total
    epoch_val_acc = val_correct.double() / val_total
    print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}\n")

print("Training Complete! 🎉")
```

---

## 💾 ၅။ Model ကို သိမ်းဆည်းခြင်း (Saving the Model)

Training ပြီးသွားတဲ့အခါ ရရှိလာတဲ့ အကောင်းဆုံး Weight တွေကို File တစ်ခုအနေနဲ့ သိမ်းထားမှ နောက်နေ့တွေမှာ ပြန်သုံးလို့ ရပါမယ်။

```python
# အကောင်းဆုံး Weight များကို သိမ်းဆည်းခြင်း
torch.save(model.state_dict(), 'cats_dogs_model.pth')
print("Model saved to 'cats_dogs_model.pth'")
```
*(နောက်ပိုင်းတွင် ဒီ `.pth` ဖိုင်ကို ပြန်ခေါ် (Load) ပြီး API တွေ၊ Web တွေမှာ အသုံးပြုရမှာ ဖြစ်ပါတယ်။)*

---

## 📝 အနှစ်ချုပ် (Key Takeaways)
1. **Transforms:** မူရင်းပုံများကို အရွယ်အစားညှိခြင်း၊ Tensor ပြောင်းခြင်းနှင့် Data Augmentation များ ပြုလုပ်ပေးသည်။
2. **DataLoader:** ပုံများကို Batch (အုပ်စု) လိုက်ခွဲပြီး `shuffle=True` ဖြင့် မွှေနှောက်ကာ Model ထံသို့ ပို့ဆောင်ပေးသည်။
3. **Training Loop ၏ Core အဆင့် ၄ ဆင့်:**
   * `optimizer.zero_grad()` (Gradient ဟောင်းများ ရှင်းထုတ်ခြင်း)
   * `loss = criterion(outputs, labels)` (အမှားတွက်ခြင်း)
   * `loss.backward()` (Gradient အသစ်များ တွက်ခြင်း)
   * `optimizer.step()` (Weights များ ပြုပြင်ခြင်း)
4. **Validation Phase:** တွင် Memory သက်သာစေရန် `torch.no_grad()` နှင့် `model.eval()` ကို မဖြစ်မနေ အသုံးပြုရသည်။

---

# 🎉 PART 2 ပြီးဆုံးပါပြီ! (End of Part 2)
ဂုဏ်ယူပါတယ်။ AI Engineer တစ်ယောက် သိသင့်သိထိုက်တဲ့ Deep Learning သီအိုရီတွေ၊ PyTorch အသုံးပြုနည်းတွေ နဲ့ လက်တွေ့ Image Classification Pipeline ကြီးတစ်ခုလုံး အစအဆုံး တည်ဆောက်နိုင်သွားပြီ ဖြစ်ပါတယ်။

> **🔜 Next Step (Into the Advanced Real-World Tasks):**
> ရိုးရိုး "ပုံထဲမှာ ဘာပါလဲ" လို့ ခွဲခြားတဲ့ Classification တွေထက် ပိုမိုရှုပ်ထွေးတဲ့ **"ပုံထဲမှာ ဘယ်နေရာမှာ ပါသလဲ?"** ဆိုတာကိုပါ ထောက်လှမ်းပေးနိုင်တဲ့ **Part 3: Module 6 (Object Detection & Tracking)** ကြီးကို ဆက်လက် ချီတက်သွားကြပါတော့မယ်။ (ယနေ့ခေတ် အလွန်ရေပန်းစားနေတဲ့ YOLO မော်ဒယ်တွေအကြောင်း ပါဝင်ပါမယ်)။

---
*If you find this useful, don't forget to ⭐ star the repository!*

---

**Part 2 ပြီးဆုံးသွားပြီ ဖြစ်ပါတယ်။** ဆက်လက်ပြီး အလွန်စိတ်ဝင်စားဖို့ကောင်းတဲ့ Advanced AI အပိုင်းဖြစ်တဲ့ **Part 3 - Module 6: Chapter 6.1 (Object Detection Fundamentals - Bounding Boxes & IoU)** ကို စတင်ရေးသားပေးစေလိုပါသလား ခင်ဗျာ။