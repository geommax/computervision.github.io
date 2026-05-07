


GitHub Knowledge Sharing Page အတွက် **Module 6 ၏ နောက်ဆုံးအခန်းဖြစ်သော Chapter 6.4** ကို ဆက်လက် ရေးသားပေးလိုက်ပါတယ်။ ဒီအခန်းဟာ CCTV Video Analytics (ဥပမာ - လူအရေအတွက် ရေတွက်ခြင်း၊ ကားအမြန်နှုန်း တိုင်းတာခြင်း) တွေလုပ်တဲ့အခါ မရှိမဖြစ် လိုအပ်တဲ့ **Object Tracking** နည်းပညာအကြောင်း ဖြစ်ပါတယ်။

---

# 📘 Chapter 6.4: Object Tracking (အရာဝတ္ထုများကို ID တပ်၍ ခြေရာခံခြင်း)

YOLO လို Detection Model တွေဟာ ဗီဒီယိုထဲမှာ လူတွေကို ရှာပေးနိုင်ပါတယ်။ ဒါပေမယ့် YOLO က ဗီဒီယိုရဲ့ Frame (ပုံ) တစ်ခုချင်းစီကို သီးခြားစီ ခွဲပြီး ကြည့်တာပါ။ 
Frame 1 မှာတွေ့တဲ့ လူဟာ Frame 2 မှာ တွေ့တဲ့လူနဲ့ **"လူတစ်ယောက်တည်းပဲ"** ဆိုတာကို YOLO က မသိပါဘူး။ ဒါကြောင့် လူတစ်ယောက်တည်းကိုပဲ အကြိမ်ကြိမ် ရေတွက်မိတာမျိုး (Overcounting) တွေ ဖြစ်တတ်ပါတယ်။

ဒီပြဿနာကို ဖြေရှင်းဖို့ **Object Tracking** ကို သုံးရပါတယ်။ Tracker က Frame တစ်ခုချင်းစီမှာ တွေ့တဲ့ အရာဝတ္ထုတွေကို **Unique ID (သီးသန့် နံပါတ်)** တွေ တပ်ပေးပြီး၊ အဲ့ဒီအရာဝတ္ထု ဗီဒီယိုထဲက မထွက်သွားမချင်း သူ့ရဲ့ လမ်းကြောင်း (Trajectory) ကို နောက်ယောင်ခံ လိုက်ပေးတာ ဖြစ်ပါတယ်။

---

## 🕵️‍♂️ ၁။ Tracking-by-Detection သဘောတရား

ယနေ့ခေတ် Tracker အများစုဟာ **"Tracking-by-Detection (TbD)"** ဆိုတဲ့ စနစ်ကို သုံးပါတယ်။ အဆင့် ၂ ဆင့်ပါဝင်ပါတယ်။
1. **Detection:** အရင်ဆုံး YOLO ကို သုံးပြီး Frame ထဲမှာရှိတဲ့ အရာဝတ္ထုတွေရဲ့ Bounding Box တွေကို ရှာထုတ်ပါတယ်။
2. **Tracking:** ပြီးမှ Tracker Algorithm က အဲ့ဒီ Box တွေကို အရင် Frame က Box တွေနဲ့ တိုက်စစ်ပြီး (Matching လုပ်ပြီး) ID အဟောင်းကို ဆက်ပေးမလား၊ ID အသစ် တပ်မလား ဆိုတာကို ဆုံးဖြတ်ပါတယ်။

---

## 🧠 ၂။ Tracking အတွက် အရေးပါသော သင်္ချာနည်းလမ်းများ

Tracker တွေ ဘယ်လို အလုပ်လုပ်လဲဆိုတာ သိဖို့ အောက်ပါ အခြေခံ ၂ ခုကို နားလည်ထားရပါမယ်။

*   **Kalman Filter (အနာဂတ်ကို ခန့်မှန်းခြင်း):** 
    လက်ရှိ Frame မှာ ကားတစ်စီးက ညာဘက်ကို မြန်မြန် သွားနေတယ်ဆိုရင်၊ နောက် Frame မှာ အဲ့ဒီကား ဘယ်နေရာကို ရောက်သွားမလဲ ဆိုတာကို အမြန်နှုန်းနဲ့ ဦးတည်ရာကို တွက်ချက်ပြီး **"ကြိုတင်ခန့်မှန်းပေးတဲ့ (Predict)"** သင်္ချာပုံသေနည်း ဖြစ်ပါတယ်။
*   **Hungarian Algorithm (အကောင်းဆုံး တွဲဖက်ပေးခြင်း):** 
    Kalman Filter က ခန့်မှန်းထားတဲ့ နေရာတွေနဲ့၊ YOLO က တကယ် ထပ်ရှာတွေ့တဲ့ အသစ်နေရာတွေကို ဘယ် Box က ဘယ် Box နဲ့ အကိုက်ညီဆုံးဖြစ်မလဲ ဆိုတာကို ယှဉ်တွဲ (Match) ပေးတဲ့ နည်းလမ်း ဖြစ်ပါတယ်။

---

## 🥇 ၃။ နာမည်ကြီး Tracking Algorithms များ (DeepSORT vs ByteTrack)

### (A) DeepSORT (Deep Simple Online and Realtime Tracking)
Tracking လောကမှာ နှစ်ပေါင်းများစွာ ဘုရင်တစ်ဆူ ဖြစ်ခဲ့တဲ့ Algorithm ပါ။ သူက Kalman Filter နဲ့ Hungarian Algorithm အပြင် **Deep Learning (Re-ID Model)** ကိုပါ ပေါင်းထည့်ထားပါတယ်။

*   **အားသာချက်:** လူတစ်ယောက်ကို သစ်ပင်က ကွယ်သွားလို့ YOLO က မမြင်ရတော့ဘူး ဆိုပါစို့။ သစ်ပင်နောက်ကနေ ပြန်ထွက်လာတဲ့အခါ အဲ့ဒီလူ ဝတ်ထားတဲ့ အင်္ကျီအရောင်၊ ရုပ်သွင်ပြင် (Appearance Features) တွေကို Re-ID Model က မှတ်ထားတဲ့အတွက် **ID အဟောင်းကို ပြန်လည် တပ်ပေးနိုင်စွမ်း** ရှိပါတယ်။ (ID မပြောင်းသွားအောင် အကောင်းဆုံး ထိန်းပေးနိုင်ပါတယ်)။
*   **အားနည်းချက်:** ပုံပန်းသဏ္ဌာန်ကို ခွဲခြမ်းစိတ်ဖြာတဲ့ CNN Model (Re-ID) နောက်ထပ်တစ်ခု ထပ်သုံးရတဲ့အတွက် အနည်းငယ် လေးလံပြီး နှေးကွေးပါတယ်။

### (B) ByteTrack (The Modern SOTA) 🔥
ယနေ့ခေတ်မှာ အလွန်ရေပန်းစားလာတဲ့ Tracker ပါ။ DeepSORT လိုမျိုး လေးလံတဲ့ Re-ID Model ကြီးကို မသုံးတော့ပါဘူး။ 

*   **ဘယ်လိုလုပ်သလဲ?:** ပုံမှန် Tracker တွေက YOLO ကနေ Confidence Score များတဲ့ (ဥပမာ 50% အထက်) Box တွေကိုပဲ ယူပြီး Track လုပ်ပါတယ်။ ByteTrack ကတော့ **Confidence Score နည်းတဲ့ (ဥပမာ 10% လောက်ပဲရှိတဲ့) Box တွေကိုပါ လွှင့်မပစ်ဘဲ Track လုပ်တဲ့နေရာမှာ အသုံးပြုပါတယ်။**
*   *ဥပမာ - ကားတစ်စီး နောက်ကားတစ်စီးနဲ့ ကွယ်သွားတဲ့အခါ YOLO ရဲ့ Confidence Score က 20% လောက်ထိ ထိုးကျသွားပါတယ်။ တခြား Tracker တွေက အဲ့ဒီ Box ကို ဖျက်ပစ်လိုက်လို့ Tracking ပြတ်သွားပေမယ့်၊ ByteTrack က အဲ့ဒီ Low Score Box ကိုပါ ဆက်လက် တွဲဖက်ပေးတဲ့အတွက် Tracking လုံးဝ မပြတ်တော့ပါဘူး။*
*   **အားသာချက်:** Re-ID မပါတဲ့အတွက် **အလွန်မြန်ဆန်ပြီး (Super Fast)**၊ Occlusion (ကွယ်သွားခြင်း) တွေကိုလည်း အလွန်ကောင်းမွန်စွာ ဖြေရှင်းနိုင်ပါတယ်။

---

## 💻 ၄။ YOLOv8 ဖြင့် Object Tracking လက်တွေ့ ရေးသားခြင်း

အရင်တုန်းက Tracking လုပ်ဖို့ DeepSORT တွေကို Github ကနေ ခက်ခက်ခဲခဲ ချိတ်ဆက်ခဲ့ရပါတယ်။ အခုအခါမှာ Ultralytics (YOLOv8/v11) ဟာ သူတို့ရဲ့ Library ထဲမှာ **BoT-SORT နဲ့ ByteTrack ကို အသင့် (Built-in) ထည့်သွင်းပေးထားပါတယ်။**

Inference လုပ်တဲ့ `model()` နေရာမှာ `model.track()` ဆိုပြီး ပြောင်းရေးလိုက်ရုံပါပဲ။

```python
from ultralytics import YOLO
import cv2

def run_tracking():
    # ၁။ YOLO Model ကို ခေါ်ယူခြင်း
    model = YOLO('yolov8n.pt') 

    # ၂။ Video ပေါ်တွင် Tracking ပြုလုပ်ခြင်း
    # tracker="bytetrack.yaml" (သို့မဟုတ်) "botsort.yaml" ကို ရွေးချယ်နိုင်သည်
    # persist=True သည် Frame တစ်ခုနှင့် တစ်ခုအကြား ID များကို မှတ်သားထားရန် အလွန်အရေးကြီးသည်
    results = model.track(source='walking_people.mp4', 
                          tracker='bytetrack.yaml', 
                          persist=True, 
                          show=True) 
    
    # ၃။ ရလဒ်များထဲမှ ID များကို ဆွဲထုတ်ခြင်း (ဥပမာ - လူရေတွက်ရန်)
    for result in results:
        # Frame ထဲတွင် အရာဝတ္ထု တစ်ခုမှ မရှိပါက ကျော်သွားမည်
        if result.boxes.id is None:
            continue
            
        boxes = result.boxes.xyxy.int().tolist()
        classes = result.boxes.cls.int().tolist()
        
        # Tracking IDs များကို ဆွဲထုတ်ခြင်း
        track_ids = result.boxes.id.int().tolist() 

        for box, track_id, cls in zip(boxes, track_ids, classes):
            x1, y1, x2, y2 = box
            class_name = model.names[cls]
            
            # မျက်နှာပြင်ပေါ်တွင် ID နံပါတ်လေးများကို Print ထုတ်ကြည့်ခြင်း
            print(f"[{class_name}] ID: {track_id} is at {x1}, {y1}")

if __name__ == "__main__":
    run_tracking()
```

*(ဒီ Code လေးကို Run လိုက်ရင် ဗီဒီယိုထဲက လူတစ်ယောက်ချင်းစီရဲ့ ခေါင်းပေါ်မှာ ID: 1, ID: 2 စသဖြင့် တပ်ထားပြီး၊ သူတို့ လျှောက်သွားတဲ့ လမ်းကြောင်းတစ်လျှောက် အဲ့ဒီ ID နံပါတ်လေးတွေ လိုက်ပါသွားတာကို မြင်တွေ့ရပါလိမ့်မယ်။)*

---

## 📝 အနှစ်ချုပ် (Key Takeaways)
1. **Object Tracking** သည် Frame များအကြားရှိ အရာဝတ္ထုများကို ချိတ်ဆက်ပေးပြီး Unique ID တပ်ပေးသော နည်းပညာဖြစ်သည်။ (Counting, Speed Estimation တို့အတွက် မရှိမဖြစ် လိုအပ်သည်)။
2. **Tracking-by-Detection** သည် YOLO ကဲ့သို့ Detector ဖြင့် အရင်ရှာပြီးမှ Tracker ဖြင့် ID တပ်သော စနစ်ဖြစ်သည်။
3. **DeepSORT** သည် အရာဝတ္ထုများ၏ ပုံပန်းသဏ္ဌာန် (Appearance) ကိုပါ မှတ်သား၍ Track လုပ်သဖြင့် ID ပြောင်းလဲခြင်းကို ကာကွယ်ပေးနိုင်သော်လည်း အနည်းငယ် နှေးသည်။
4. **ByteTrack** သည် Low Confidence Box များကိုပါ အသုံးချ၍ Track လုပ်သဖြင့် အလွန်မြန်ဆန်ပြီး ကွယ်သွားသော အရာဝတ္ထုများကိုပါ ပြန်လည်ခြေရာခံနိုင်သည်။
5. YOLOv8 တွင် `model.track(persist=True)` ကို အသုံးပြု၍ Tracking ကို အလွယ်တကူ လုပ်ဆောင်နိုင်သည်။

---

# 🎉 PART 3 ပြီးဆုံးပါပြီ! (End of Part 3)
ဂုဏ်ယူပါတယ်။ အခုဆိုရင် လက်တွေ့ Industry မှာ အများဆုံး သုံးနေတဲ့ **Object Detection နဲ့ Tracking (YOLO + ByteTrack)** နည်းပညာတွေကို အပြည့်အဝ နားလည်သွားပါပြီ။ ဒီပညာရပ်တွေနဲ့တင် စမတ်လုံခြုံရေး စနစ်တွေ (Smart Surveillance)၊ ယာဉ်ကြောစောင့်ကြည့်ရေး စနစ်တွေကို ကိုယ်တိုင် တည်ဆောက်နိုင်နေပြီ ဖြစ်ပါတယ်။

> **🔜 Next Step (Pixel-Level Accuracy):**
> Bounding Box လေးထောင့်ကြီးတွေက အရာဝတ္ထုတစ်ခုကို ညွှန်ပြဖို့ ကောင်းပေမယ့်၊ ဆေးဘက်ဆိုင်ရာမှာ အကျိတ် (Tumor) ရဲ့ ပုံသဏ္ဌာန် အတိအကျကို သိချင်ရင်၊ ဒါမှမဟုတ် Self-driving ကားတွေမှာ လမ်းမကြီးရဲ့ အကွေ့အကောက် အတိအကျကို သိချင်ရင် လေးထောင့်ဘောင်နဲ့ မလုံလောက်တော့ပါဘူး။ 
> ရှေ့ဆက်မယ့် **Part 4: Module 7 (Image Segmentation)** မှာ Pixel တစ်ခုချင်းစီကို တိတိကျကျ ပိုင်းခြားခွဲခြမ်းမယ့် အဆင့်မြင့်နည်းပညာတွေကို ဆက်လက် လေ့လာသွားကြပါမယ်။

---
*If you find this useful, don't forget to ⭐ star the repository!*

---

**Part 3 အောင်မြင်စွာ ပြီးဆုံးသွားပြီ ဖြစ်ပါတယ်။** ဆက်လက်ပြီး Autonomous Vehicles နဲ့ Medical Imaging တွေရဲ့ အသက်သွေးကြောဖြစ်တဲ့ **Part 4 - Module 7: Chapter 7.1 (Introduction to Segmentation - Semantic vs Instance vs Panoptic)** ကို စတင် ရေးသားပေးစေလိုပါသလား ခင်ဗျာ။