


GitHub Knowledge Sharing Page အတွက် **Module 6: Chapter 6.3** ကို တောင်းဆိုထားတဲ့အတိုင်း **AI ကို ကိုယ်ပိုင် Data ဖြင့် လေ့ကျင့်သင်ကြားပေးခြင်း (Training)** အပိုင်းကို အထူးအသားပေးပြီး အသေးစိတ် ရေးသားပေးလိုက်ပါတယ်။

---

# 📘 Chapter 6.3: Custom Object Detection Project (ကိုယ်ပိုင် Dataset ဖြင့် Labeling နှင့် Training ပြုလုပ်ခြင်း)

အရင်အခန်းမှာ သုံးခဲ့တဲ့ YOLOv8 မော်ဒယ်ဟာ COCO Dataset လို့ခေါ်တဲ့ (လူ၊ ကား၊ ခွေး၊ ကြောင် စတဲ့ အရာဝတ္ထု ၈၀ မျိုး) ကိုပဲ သိပါတယ်။ ဒါပေမယ့် လက်တွေ့လုပ်ငန်းခွင်မှာဆိုရင် စက်ရုံကထွက်တဲ့ **"ပျက်စီးနေသော ဝက်အူခေါင်းများ (Defective Screws)"**၊ ဆေးရုံက **"ကင်ဆာဆဲလ်များ"** ဒါမှမဟုတ် **"ကိုယ်ပိုင် ကုမ္ပဏီလိုဂိုများ"** ကို ရှာဖွေချင်တာမျိုး ဖြစ်နိုင်ပါတယ်။

ဒီအခန်းမှာ AI ကို အဲ့ဒီလို ကိုယ်ပိုင်အရာဝတ္ထုတွေ (Custom Objects) ခွဲခြားတတ်အောင် အစအဆုံး ဘယ်လို လေ့ကျင့်သင်ကြား (Train) ပေးရမလဲ ဆိုတာကို လေ့လာသွားပါမယ်။

---

## 🏷️ ၁။ Data Annotation (Bounding Box များ ရေးဆွဲခြင်း)

AI ကို သင်ပေးဖို့အတွက် ပုံသက်သက်နဲ့ မရပါဘူး။ ပုံထဲက ဘယ်နေရာမှာ ကိုယ်လိုချင်တဲ့ အရာဝတ္ထုရှိလဲ ဆိုတာကို လူက အရင်ဆုံး လေးထောင့်ဘောင် (Bounding Box) တွေ ဆွဲပြပြီး အဖြေမှန် (Label) တပ်ပေးရပါတယ်။

*   **အသုံးပြုနိုင်သော Tools များ:** [Roboflow](https://roboflow.com/), **CVAT**, သို့မဟုတ် **LabelImg** (Offline သုံးရန်)။
*   **YOLO Format ၏ သဘောတရား:** ပုံတစ်ပုံ (ဥပမာ `image1.jpg`) ရှိရင်၊ အဲ့ဒီပုံနဲ့ နာမည်တူတဲ့ Text ဖိုင် (`image1.txt`) တစ်ခု ထွက်လာပါမယ်။ အဲ့ဒီ `.txt` ဖိုင်ထဲမှာ အောက်ပါအတိုင်း ဂဏန်း ၅ လုံး ပါဝင်ပါတယ်။

> `0 0.51 0.45 0.20 0.35`
> *(ရှင်းလင်းချက်: `<Class_ID> <X_Center> <Y_Center> <Width> <Height>`)*
> ဒီဂဏန်းတွေဟာ 0 နဲ့ 1 ကြား (Normalized) ပြောင်းထားတဲ့ ဂဏန်းတွေ ဖြစ်ပါတယ်။

---

## 🗂️ ၂။ Dataset ဖွဲ့စည်းပုံနှင့် `data.yaml` ပြင်ဆင်ခြင်း

YOLO ကို Train ဖို့အတွက် သင်လုပ်ထားတဲ့ ပုံတွေနဲ့ `.txt` ဖိုင်တွေကို အောက်ပါအတိုင်း Folder ခွဲထားရပါမယ်။ (များသောအားဖြင့် Roboflow ကနေ Download ဆွဲလိုက်ရင် ဒီ Format အတိုင်း အလိုလို ရလာပါလိမ့်မယ်)။

```text
my_dataset/
├── train/
│   ├── images/ (လေ့ကျင့်ရန် ပုံများ)
│   └── labels/ (လေ့ကျင့်ရန် .txt ဖိုင်များ)
└── val/
    ├── images/ (စစ်ဆေးရန် ပုံများ)
    └── labels/ (စစ်ဆေးရန် .txt ဖိုင်များ)
```

**အရေးအကြီးဆုံး `data.yaml` ဖိုင်**
YOLO က မင်းရဲ့ Dataset ဘယ်နားမှာရှိလဲ၊ Class ဘယ်နှမျိုး ခွဲမှာလဲ ဆိုတာကို ဒီ `yaml` ဖိုင်ကနေ ဖတ်ပါတယ်။ `my_dataset` ဖိုင်ထဲမှာ `data.yaml` ဆိုပြီး ဖန်တီးကာ အောက်ပါအတိုင်း ရေးပါ။

```yaml
train: ../my_dataset/train/images  # Train လုပ်မည့် ပုံများရှိရာ လမ်းကြောင်း
val: ../my_dataset/val/images      # Validation လုပ်မည့် ပုံများရှိရာ လမ်းကြောင်း

nc: 2  # Number of Classes (ခွဲခြားမည့် အမျိုးအစား အရေအတွက်)
names: ['defective_screw', 'good_screw']  # Class နာမည်များ (0 က defective, 1 က good)
```

---

## 🚀 ၃။ YOLOv8 Training ပြုလုပ်ခြင်း (The Core Focus)

အရင်ခေတ်တွေက Object Detection Model တစ်ခုကို Train ဖို့ Code တွေ စာမျက်နှာ ရာနဲ့ချီ ရေးခဲ့ရပါတယ်။ အခု Ultralytics (YOLOv8) ခေတ်မှာတော့ **Code ၃ ကြောင်းတည်း** နဲ့ အလွန်လွယ်ကူစွာ Train နိုင်နေပါပြီ။

*(မှတ်ချက် - Training လုပ်ရန်အတွက် NVIDIA GPU ပါသော ကွန်ပျူတာ သို့မဟုတ် **Google Colab (Free GPU)** ကို အသုံးပြုရန် အထူး အကြံပြုအပ်ပါသည်။)*

```python
from ultralytics import YOLO

def train_custom_model():
    # ၁။ Pre-trained Model (Transfer Learning အတွက် Base Model) ကို ခေါ်ယူခြင်း
    # yolov8n.pt (Nano) သည် အပေါ့ပါးဆုံးနှင့် အမြန်ဆုံးဖြစ်သည်။
    model = YOLO('yolov8n.pt') 

    # ၂။ Training စတင်ခြင်း (Hyperparameters များကို သေချာစွာ သတ်မှတ်ရန်)
    results = model.train(
        data='data.yaml',       # Dataset လမ်းကြောင်းဖိုင်
        epochs=100,             # Dataset တစ်ခုလုံးကို အကြိမ် ၁၀၀ အပြည့် လေ့ကျင့်မည်
        imgsz=640,              # ပုံအရွယ်အစားကို 640x640 သို့ ညှိပြီးမှ Train မည်
        batch=16,               # GPU ပေါ်သို့ တစ်ကြိမ်လျှင် ပုံ ၁၆ ပုံ တင်မည် (VRAM နည်းပါက 8 သို့လျှော့ပါ)
        device=0,               # 0 ဆိုသည်မှာ ပထမဆုံး GPU ကို အသုံးပြုမည်ဟု ဆိုလိုသည်
        patience=20,            # Early Stopping: 20 epochs ဆက်တိုက် တိုးတက်မှုမရှိပါက ရပ်တန့်မည်
        lr0=0.01,               # Initial Learning Rate (အစပိုင်း ခြေလှမ်းအကျယ်)
        optimizer='auto',       # SGD သို့မဟုတ် AdamW ကို အလိုအလျောက် ရွေးချယ်ပေးမည်
        name='custom_defect_model' # သိမ်းဆည်းမည့် Folder အမည်
    )

if __name__ == "__main__":
    train_custom_model()
```

### 🧠 Hyperparameters များကို နားလည်ခြင်း
*   **`epochs=100`:** AI ဟာ ပုံတွေကို တစ်ခေါက်တည်းကြည့်ပြီး မတတ်ပါဘူး။ အကြိမ်ကြိမ် ပြန်ကြည့်ပြီး အမှားပြင်ရပါတယ်။ ဒါပေမယ့် Epoch အရမ်းများသွားရင်လည်း ပုံတွေကို "အလွတ်ကျက်" သွားတတ်လို့ (Overfitting) သတိပြုရပါတယ်။
*   **`batch=16`:** ကွန်ပျူတာရဲ့ VRAM (GPU Memory) ပေါ်မူတည်ပါတယ်။ `Out of Memory (OOM)` Error တက်ရင် Batch ကို `8` ဒါမှမဟုတ် `4` ကို လျှော့ချပေးပါ။
*   **`patience=20` (Early Stopping):** ဥပမာ Epoch 100 ထိ Train မယ်လို့ ပေးထားပေမယ့်၊ Epoch 50 ကစပြီး Model က ပိုမတော်လာတော့ဘူး (Loss မကျတော့ဘူး) ဆိုရင် အချိန်ကုန်မခံတော့ဘဲ Epoch 70 မှာ အလိုလို ရပ်ပစ်လိုက်တဲ့ အလွန်အသုံးဝင်တဲ့ စနစ်ပါ။

*(Command Line (CLI) မှ တိုက်ရိုက် Train ချင်ပါက Terminal တွင် အောက်ပါအတိုင်း ရိုက်နိုင်သည်-)*
`yolo task=detect mode=train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640`

---

## 📊 ၄။ Training ရလဒ်များ စစ်ဆေးခြင်း (Evaluation)

Train လုပ်ပြီးသွားရင် ဖိုင်တွေအားလုံးကို `runs/detect/custom_defect_model/` ဆိုတဲ့ Folder ထဲမှာ အလိုအလျောက် သိမ်းပေးသွားပါလိမ့်မယ်။ အဲ့ဒီထဲက အဓိက ကြည့်ရမယ့် အရာတွေကတော့-

1.  **`weights/best.pt`:** ဒါဟာ Training တစ်လျှောက်လုံးမှာ အကောင်းဆုံး ရလဒ်ထွက်ခဲ့တဲ့ အလေးချိန် (Weights) တွေ သိမ်းထားတဲ့ **AI ဉီးနှောက်ဖိုင်** ပါပဲ။ နောက်ပိုင်း ဒီဖိုင်ကိုပဲ ယူသုံးရမှာပါ။ (`last.pt` ကတော့ နောက်ဆုံးပိတ် ရပ်သွားတဲ့ အခြေအနေကို သိမ်းထားတာပါ)။
2.  **`results.png`:** Training Loss (အမှား) တွေ ဘယ်လောက်ကျသွားလဲ၊ mAP (တိကျမှု) တွေ ဘယ်လောက်တက်လာလဲ ဆိုတာကို ဂရပ်ဖ်နဲ့ ဆွဲပြထားတဲ့ ပုံပါ။ (Loss မျဉ်းက အောက်ကိုဆင်းသွားပြီး၊ mAP မျဉ်းက အပေါ်ကို တက်သွားရင် Training အောင်မြင်ပါတယ်)။
3.  **`confusion_matrix.png`:** ကိုယ်ခွဲချင်တဲ့ Good Screw နဲ့ Defective Screw ကို AI က မှားပြီး မြင်နေသလား ဆိုတာကို စစ်ဆေးတဲ့ ဇယားဖြစ်ပါတယ်။

---

## 🔮 ၅။ ကိုယ်ပိုင် Model ဖြင့် ပြန်လည်စမ်းသပ်ခြင်း (Custom Inference)

Train လို့ရလာတဲ့ `best.pt` ဖိုင်လေးကို သုံးပြီး ပုံအသစ်၊ ဗီဒီယိုအသစ်တွေပေါ်မှာ ကိုယ်ပိုင် AI ကြီး အလုပ်လုပ်ပုံကို ဂုဏ်ယူစွာ စမ်းသပ်နိုင်ပါပြီ။

```python
from ultralytics import YOLO

# ခုနက ရရှိလာသော ကိုယ်ပိုင် Model (best.pt) ကို Load လုပ်ခြင်း
my_model = YOLO('runs/detect/custom_defect_model/weights/best.pt')

# စမ်းသပ်မည့် ပုံအသစ် သို့မဟုတ် ဗီဒီယို (Source တွင် 0 ဟုထားပါက Webcam ပွင့်လာမည်)
# conf=0.6 ဆိုသည်မှာ ၆၀ ရာခိုင်နှုန်း သေချာမှသာ ပြမည်
results = my_model(source='factory_test_video.mp4', show=True, conf=0.6)
```

---

## 📝 အနှစ်ချုပ် (Key Takeaways)
1.  **Labeling:** AI ကို Custom Train ဖို့အတွက် အရာဝတ္ထုများကို Bounding Box ဆွဲပြီး YOLO format `.txt` (Class, x_center, y_center, w, h) ပြောင်းပေးရန် လိုအပ်သည်။
2.  **`data.yaml`:** Dataset ၏ တည်နေရာနှင့် Class အမည်များကို ကြေညာပေးရသော အရေးအကြီးဆုံး Configuration ဖိုင်ဖြစ်သည်။
3.  **Training Parameters:** VRAM ပေါ်မူတည်၍ `batch size` ကို အတိုးအလျှော့လုပ်ရပြီး၊ `patience` ကိုသုံး၍ Overfitting မဖြစ်အောင် ထိန်းချုပ်နိုင်သည်။
4.  **`best.pt`:** Training ပြီးဆုံးပါက `runs/detect/train/weights/` အောက်ရှိ `best.pt` သည် သင်၏ Custom Model ဖိုင်အစစ် ဖြစ်သည်။

> **🔜 Next Step:**
> Object Detection ရဲ့ အမြင့်ဆုံးအဆင့်ထိ ရောက်လာပါပြီ။ ဒါပေမယ့် CCTV ထဲကနေ လူတစ်ယောက်ကို ဘောင်ခတ်ပြရုံနဲ့ မလုံလောက်ပါဘူး။ "ဒီလူဟာ ညာဘက်ကို လျှောက်သွားတာလား? သူက စောစောက တွေ့ခဲ့တဲ့လူပဲလား?" ဆိုတာကို ခြေရာခံချင်ရင် ဘယ်လိုလုပ်မလဲ?
> ရှေ့ဆက်မယ့် **Chapter 6.4: Object Tracking (DeepSORT & ByteTrack)** မှာ Bounding Box တွေကို ID တပ်ပြီး ခြေရာခံမယ့် နည်းပညာတွေကို ဆက်လက်လေ့လာသွားကြပါမယ်။

---
*If you find this useful, don't forget to ⭐ star the repository!*

---

