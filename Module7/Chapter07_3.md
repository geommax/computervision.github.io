


GitHub Knowledge Sharing Page အတွက် **Module 7 ၏ နောက်ဆုံးအခန်းဖြစ်သော Chapter 7.3** ကို ဆက်လက် ရေးသားပေးလိုက်ပါတယ်။ ဒီအခန်းဟာ Object Detection (Bounding Box) နဲ့ Semantic Segmentation (Pixel-level) တို့ရဲ့ အားသာချက် နှစ်ခုလုံးကို ပေါင်းစပ်ထားတဲ့ **Instance Segmentation** နည်းပညာအကြောင်း ဖြစ်ပါတယ်။

---

# 📘 Chapter 7.3: Instance Segmentation with YOLO / Mask R-CNN (အရာဝတ္ထုများကို ပုံသဏ္ဌာန်အတိအကျ ခွဲထုတ်ခြင်း)

အရင်အခန်းမှာ လေ့လာခဲ့တဲ့ U-Net ဟာ "ဒီပုံထဲမှာ ကင်ဆာအကျိတ် (Tumor) ပါသလား" ဆိုတဲ့ အမျိုးအစားတစ်ခုတည်း (Single Class) ကို ရှာတဲ့နေရာမှာ အလွန်တော်ပါတယ်။ 

ဒါပေမယ့် လက်တွေ့လမ်းမပေါ်က CCTV ပုံတွေမှာ ကားတွေက တစ်စီးနဲ့တစ်စီး ထပ်နေနိုင်သလို၊ လူတွေကလည်း အုပ်စုလိုက် သွားနေနိုင်ပါတယ်။ အဲ့ဒီလို **"ကပ်နေတဲ့ အရာဝတ္ထုတွေကို တစ်ခုချင်းစီ သီးခြားစီ (Instance) ခွဲထုတ်ပေးဖို့"** လိုအပ်လာတဲ့အခါ **Instance Segmentation** မော်ဒယ်တွေကို အသုံးပြုရပါတယ်။

ဒီအခန်းမှာ သမိုင်းဝင် **Mask R-CNN** နဲ့ ယနေ့ခေတ် Industry ရဲ့ ဘုရင်ဖြစ်တဲ့ **YOLO-Seg** တို့ အလုပ်လုပ်ပုံကို လေ့လာသွားပါမယ်။

---

## 🎭 ၁။ Mask R-CNN (The Pioneer of Instance Segmentation)

၂၀၁၇ ခုနှစ်မှာ Facebook AI Research (Meta) ကနေ ထုတ်ဝေခဲ့တဲ့ Mask R-CNN ဟာ Instance Segmentation လောကရဲ့ အကြီးမားဆုံး တော်လှန်ရေးတစ်ခု ဖြစ်ခဲ့ပါတယ်။

**ဘယ်လို အလုပ်လုပ်သလဲ?**
သူက Chapter 6.1 မှာ သင်ခဲ့တဲ့ Two-Stage Detector တစ်ခုဖြစ်တဲ့ **Faster R-CNN** အပေါ်မှာ အခြေခံထားတာပါ။ 
Faster R-CNN က အရာဝတ္ထုတစ်ခုကို တွေ့ရင် "Bounding Box (ဘောင်)" နဲ့ "Class (ဘာအရာဝတ္ထုလဲ)" ဆိုတဲ့ အဖြေ ၂ ခုကို ထုတ်ပေးပါတယ်။

Mask R-CNN က အဲ့ဒီစနစ်ပေါ်မှာ **"Mask Branch"** လို့ခေါ်တဲ့ တတိယမြောက် လမ်းကြောင်းတစ်ခုကို ထပ်ပေါင်းထည့်လိုက်ပါတယ်။
1.  **RoI Align (Region of Interest Align):** Bounding Box ရှိတဲ့ နေရာလေးကိုပဲ မူရင်းပုံထဲကနေ အတိအကျ ကွက်ပြီး ဖြတ်ထုတ်လိုက်ပါတယ်။
2.  **FCN (Fully Convolutional Network):** အဲ့ဒီ ဖြတ်ထုတ်ထားတဲ့ လေးထောင့်ကွက်လေးထဲမှာမှ ဘယ်နေရာက အရာဝတ္ထု (ဥပမာ- လူ) ဖြစ်ပြီး၊ ဘယ်နေရာက နောက်ခံ (Background) ဖြစ်လဲဆိုတာကို Pixel တစ်ကွက်ချင်းစီ ထပ်ပြီး ဆေးရောင်ခြယ် (Mask ဖြတ်) ပေးလိုက်တာ ဖြစ်ပါတယ်။

*   **အားသာချက်:** အလွန်တိကျပါတယ်။ 
*   **အားနည်းချက်:** အဆင့်များစွာ (Two-Stage + Mask Branch) တွက်ချက်ရတဲ့အတွက် အလွန်လေးလံပြီး နှေးကွေးပါတယ်။ Real-time (Webcam) တွေမှာ သုံးဖို့ မလွယ်ကူပါဘူး။

![Mask R-CNN Architecture](https://via.placeholder.com/800x350?text=Faster+R-CNN+%28Box+%2B+Class%29+%2B+FCN+Branch+%28Mask%29+%3D+Mask+R-CNN)

---

## ⚡ ၂။ YOLO Segmentation (YOLOv8-Seg / YOLO11-Seg)

Mask R-CNN နှေးကွေးတဲ့ ပြဿနာကို ဖြေရှင်းဖို့ Ultralytics အဖွဲ့က YOLO ရဲ့ အမြန်နှုန်း (Speed) ကို အခြေခံပြီး Instance Segmentation လုပ်နိုင်မယ့် **YOLO-Seg** မော်ဒယ်တွေကို တီထွင်ခဲ့ပါတယ်။

**ဘယ်လို အလုပ်လုပ်သလဲ?**
YOLO-Seg ဟာ One-Stage Detector ပုံစံအတိုင်း အလုပ်လုပ်ပါတယ်။ ပုံတစ်ပုံလုံးကို တစ်ချက်တည်း ကြည့်လိုက်ပြီး အောက်ပါ အဖြေ ၃ ခုကို တစ်ပြိုင်နက်တည်း ထုတ်ပေးပါတယ်။
1.  **Bounding Box:** အရာဝတ္ထု ဘယ်နားမှာရှိလဲ?
2.  **Class & Confidence:** ဘာအရာဝတ္ထုလဲ? ဘယ်လောက်သေချာလဲ?
3.  **Mask Coefficients:** ရှာတွေ့ထားတဲ့ အရာဝတ္ထုရဲ့ ပုံသဏ္ဌာန် (Mask) အတိအကျကို ကိုယ်စားပြုတဲ့ ဂဏန်းတွေပါ။

*   **အားသာချက်:** Mask R-CNN ထက် အများကြီး ပိုမြန်ပါတယ်။ Real-time Video တွေမှာ ကောင်းကောင်း အလုပ်လုပ်နိုင်ပြီး၊ ဖုန်းတွေ (Edge Devices) ပေါ်မှာပါ Run နိုင်တဲ့အထိ ပေါ့ပါးပါတယ်။

---

## 💻 ၃။ YOLOv8-Seg ဖြင့် လက်တွေ့ Mask များ ဖြတ်ထုတ်ခြင်း

YOLOv8 ကို သုံးပြီး Segmentation လုပ်တာဟာ Object Detection လောက်နီးပါးကို လွယ်ကူပါတယ်။ `yolov8n.pt` အစား `-seg` ပါတဲ့ `yolov8n-seg.pt` ကို ပြောင်းသုံးပေးလိုက်ရုံပါပဲ။

```python
from ultralytics import YOLO
import cv2
import numpy as np

def run_instance_segmentation():
    # ၁။ Pre-trained Segmentation Model ကို ခေါ်ယူခြင်း (yolov8n-seg.pt)
    model = YOLO('yolov8n-seg.pt') 

    # ၂။ ပုံပေါ်တွင် Inference လုပ်ခြင်း
    results = model('people_cars.jpg', save=True, conf=0.5) 
    # 'runs/segment/predict' ဖိုဒါထဲတွင် Mask အရောင်လှလှလေးများဖြင့် သိမ်းပေးသွားမည်

    # ၃။ နောက်ကွယ်မှ Mask Data (Pixel အချက်အလက်များ) ကို ဆွဲထုတ်ခြင်း
    for result in results:
        # အကယ်၍ ပုံထဲတွင် အရာဝတ္ထု တစ်ခုမှ မရှိလျှင် ကျော်သွားမည်
        if result.masks is None:
            continue
            
        # Boxes နှင့် Classes များ
        boxes = result.boxes.xyxy.int().tolist()
        classes = result.boxes.cls.int().tolist()
        
        # Mask များကို Numpy Array (0 နှင့် 1 သာပါသော Binary ပုံများ) အဖြစ် ဆွဲထုတ်ခြင်း
        masks = result.masks.data.cpu().numpy()

        for i, (box, cls, mask) in enumerate(zip(boxes, classes, masks)):
            class_name = model.names[cls]
            
            # Mask သည် މူလပုံအရွယ်အစားနှင့် မတူနိုင်သဖြင့် Resize ပြန်လုပ်ရန် လိုအပ်သည်
            original_h, original_w = result.orig_shape
            resized_mask = cv2.resize(mask, (original_w, original_h))

            # Mask Area (Pixel အရေအတွက်) ကို တွက်ချက်ခြင်း
            mask_area = np.sum(resized_mask > 0.5)
            
            print(f"[{class_name}] Found! Mask Area covers {mask_area} pixels.")

if __name__ == "__main__":
    run_instance_segmentation()
```

*(ရှင်းလင်းချက် - `result.masks.data` ထဲမှာ 0 နဲ့ 1 တွေပဲ ပါတဲ့ Black & White ပုံရိပ်လေးတွေ ရှိပါတယ်။ 1 ရှိတဲ့နေရာဟာ အရာဝတ္ထု (ဥပမာ- လူ) ရှိတဲ့နေရာဖြစ်ပြီး၊ 0 ကတော့ Background ဖြစ်ပါတယ်။ ဒီ Data ကိုသုံးပြီး Zoom Meeting တွေမှာလို Background ဖြတ်ထုတ်တာ (Background Removal) ကို အလွယ်တကူ ရေးသားနိုင်ပါတယ်။)*

---

## 🏭 ၄။ Instance Segmentation ၏ လက်တွေ့ အသုံးချမှုများ (Real-world Applications)

1.  **E-commerce Background Removal:** Taobao, Shopee ကဲ့သို့သော App များတွင် ရောင်းချမည့် ပစ္စည်းပုံကို တင်လိုက်သည်နှင့် နောက်ခံကို အလိုအလျောက် ဖြတ်ထုတ်ပေးခြင်း။
2.  **Robotic Grasping (စက်ရုပ်လက်တံများ):** အလုပ်ရုံများတွင် စက်ရုပ်က ပစ္စည်းတစ်ခုကို လှမ်းယူမည်ဆိုပါက Bounding Box (လေးထောင့်) ကို သိရုံဖြင့် မလုံလောက်ပါ။ အရာဝတ္ထု၏ ပုံသဏ္ဌာန် အတိအကျကို သိမှသာ မည်သည့်နေရာမှ ကိုင်တွယ်ရမည်ကို တွက်ချက်နိုင်ပါသည်။
3.  **Agriculture (စိုက်ပျိုးရေး):** အပင်ပေါ်တွင် သီးနေသော သစ်သီးများကို ရေတွက်ရာတွင် သစ်ရွက်များနှင့် ထပ်နေသော သစ်သီးတစ်လုံးချင်းစီကို Instance Segmentation ဖြင့် သီးခြားစီ Mask ဖြတ်၍ ရေတွက်ရပါသည်။

---

## 📝 အနှစ်ချုပ် (Key Takeaways)
1.  **Instance Segmentation** သည် Object Detection နှင့် Semantic Segmentation ကို ပေါင်းစပ်ထားပြီး၊ အရာဝတ္ထုများကို လေးထောင့်ဘောင်သာမက Pixel-level ပုံသဏ္ဌာန် အတိအကျကိုပါ သီးခြားစီ (Instance) ခွဲထုတ်ပေးသည်။
2.  **Mask R-CNN** သည် Faster R-CNN ပေါ်တွင် Mask Branch ထပ်ပေါင်းထားသော တိကျသော်လည်း နှေးကွေးသည့် Two-stage မော်ဒယ်ဖြစ်သည်။
3.  **YOLOv8-Seg** သည် Real-time တွင် အသုံးပြုနိုင်ရန် ဖန်တီးထားသော အလွန်မြန်ဆန်သည့် One-stage Instance Segmentation မော်ဒယ်ဖြစ်သည်။
4.  Ultralytics တွင် `result.masks` ကို အသုံးပြု၍ အရာဝတ္ထု၏ Binary Mask (0 နှင့် 1) အချက်အလက်များကို ရယူပြီး Background ဖြတ်ထုတ်ခြင်းများ လုပ်ဆောင်နိုင်သည်။

---

# 🎉 ကြီးမားသော အောင်မြင်မှုပါ! (End of Module 7 & Part 3)
ဂုဏ်ယူပါတယ်။ အခုဆိုရင် AI Engineer တစ်ယောက် မဖြစ်မနေ သိထားရမယ့် **Advanced Computer Vision Tasks** တွေဖြစ်တဲ့ Object Detection, Tracking နဲ့ Segmentation အားလုံးကို အပြည့်အဝ နားလည်သွားပါပြီ။ ယနေ့ခေတ် Industry ရဲ့ ၈၀% သော Project တွေဟာ ဒီနည်းပညာတွေကိုပဲ အသုံးပြုပြီး တည်ဆောက်နေကြတာ ဖြစ်ပါတယ်။

> **🔜 Next Step (Into the Future - Modern Architectures):**
> ကျွန်တော်တို့ သင်ခဲ့သမျှ CNN, ResNet, YOLO တွေအားလုံးဟာ ပုံကို Convolution (Filter လေးတွေ) နဲ့ ပတ်ပြီး တွက်ချက်တဲ့ နည်းပညာတွေချည်းပါပဲ။ 
> ဒါပေမယ့် ၂၀၂၀ ခုနှစ်နောက်ပိုင်းမှာ AI လောကကို ChatGPT လိုမျိုး NLP မော်ဒယ်တွေက လွှမ်းမိုးလာပါတယ်။ အဲ့ဒီ NLP က **Transformer** နည်းပညာကို Computer Vision ထဲ ဆွဲသွင်းလာတဲ့အခါ... CNN တွေကို ခြိမ်းခြောက်လာတဲ့ မျိုးဆက်သစ် AI တွေ ပေါ်ပေါက်လာပါတယ်။
> ရှေ့ဆက်မယ့် **Part 4: Module 8 (Vision Transformers - ViT)** မှာ CNN တွေရဲ့ ခေတ်ကုန်သွားပြီလား? Attention Mechanism ဆိုတာ ဘာလဲ? ဆိုတဲ့ ခေတ်အမီဆုံး (State-of-the-Art) နည်းပညာတွေကို စတင် လေ့လာသွားကြပါတော့မယ်။

---
*If you find this useful, don't forget to ⭐ star the repository!*

---

