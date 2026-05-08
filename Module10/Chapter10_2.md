


GitHub Knowledge Sharing Page အတွက် **Module 10 ၏ Chapter 10.2** ကို ဆက်လက် ရေးသားပေးလိုက်ပါတယ်။ ဒီအခန်းဟာ အင်တာနက်မလိုဘဲ ကင်မရာလေးတွေ၊ စက်ရုပ်လေးတွေထဲမှာ AI ကို တိုက်ရိုက်ထည့်သွင်းမောင်းနှင်တဲ့ **Edge AI** နည်းပညာအကြောင်း ဖြစ်ပါတယ်။

---

# 📘 Chapter 10.2: Deploying on Edge Devices (Edge AI နှင့် Hardware အသေးစားများပေါ်တွင် အသုံးပြုခြင်း)

ကျွန်တော်တို့ AI Model တွေကို Train တဲ့အခါ စွမ်းအားမြင့်မားတဲ့ NVIDIA RTX 4090 လို GPU ကြီးတွေ၊ ဒါမှမဟုတ် Cloud (Google Colab, AWS) ပေါ်က Supercomputer ကြီးတွေကို အသုံးပြုကြပါတယ်။ 

ဒါပေမယ့် လက်တွေ့ကမ္ဘာမှာ လမ်းဆုံက CCTV ကင်မရာ၊ စိုက်ပျိုးရေးသုံး ဒရုန်း (Drone) နဲ့ စက်ရုံက စက်ရုပ်လက်တံလေးတွေထဲကို အဲ့ဒီ ကွန်ပျူတာအကြီးကြီးတွေ သွားထည့်ထားလို့ မရပါဘူး။ ဒီလိုအချိန်မှာ ပေါ့ပါးသေးငယ်တဲ့ စက် (Hardware) လေးတွေပေါ်မှာ AI ကို တိုက်ရိုက် Run စေတဲ့ နည်းပညာကို **Edge AI (သို့မဟုတ် Edge Computing)** လို့ ခေါ်ပါတယ်။

---

## ☁️ ၁။ Cloud AI နဲ့ Edge AI ဘာကွာသလဲ? (ဘာကြောင့် Edge ကို သုံးရတာလဲ?)

"ပုံတွေကို အင်တာနက်ကနေ Cloud ပေါ်လှမ်းပို့၊ Server ကြီးက AI နဲ့ တွက်ပြီး အဖြေပြန်ပို့ပေးရင် မလွယ်ဘူးလား?" လို့ မေးစရာရှိပါတယ်။ အချို့နေရာတွေမှာ လွယ်ပေမယ့်၊ အောက်ပါ ပြဿနာ ၄ ခုကြောင့် Edge AI ကို မဖြစ်မနေ သုံးရပါတယ်။

1.  **Latency (နှောင့်နှေးမှု):** ကားအလိုအလျောက် မောင်းတဲ့စနစ်မှာ (Self-driving Car) အရှေ့က လူဖြတ်ကူးတာကို Cloud ဆီ ပို့ပြီး အဖြေပြန်စောင့်နေရင် အချိန်မမီတော့ဘဲ တိုက်မိသွားပါလိမ့်မယ်။ Edge AI က ကားပေါ်မှာတင် ချက်ချင်း (Real-time) ဆုံးဖြတ်ပါတယ်။
2.  **Bandwidth (အင်တာနက် ကုန်ကျစရိတ်):** စက်ရုံမှာရှိတဲ့ ကင်မရာ ၅၀ လုံးကနေ 4K Video တွေကို 24/7 Cloud ပေါ် အင်တာနက်ကနေ လှမ်းပို့နေမယ်ဆိုရင် Data ကုန်ကျစရိတ် အလွန်ကြီးမားပါလိမ့်မယ်။ 
3.  **Privacy (လုံခြုံရေး):** မျက်နှာဖတ်တဲ့ စနစ်တွေ၊ ဆေးရုံက လူနာအချက်အလက်တွေကို အင်တာနက်ပေါ် ပို့စရာမလိုဘဲ Local စက်ထဲမှာတင် တွက်ချက်ဖျက်ဆီးပစ်တဲ့အတွက် Data ပေါက်ကြားမှု လုံးဝ မရှိနိုင်တော့ပါဘူး။
4.  **Offline (အင်တာနက်မလိုခြင်း):** တောတောင်တွေထဲက တောရိုင်းတိရစ္ဆာန် စောင့်ကြည့်ရေး ကင်မရာတွေမှာ အင်တာနက် လုံးဝမရှိရင်တောင် Edge AI က ဆက်လက် အလုပ်လုပ်နေနိုင်ပါတယ်။

---

## 📟 ၂။ Computer Vision အတွက် နာမည်ကြီး Edge Devices များ

စက်လေးတွေက သေးငယ်ပေမယ့် AI တွက်ချက်ဖို့ အထူးဖန်တီးထားတာ ဖြစ်ပါတယ်။ အသုံးအများဆုံး ၃ မျိုးကို လေ့လာကြည့်ပါမယ်။

### (A) Raspberry Pi (ရက်စ်ဘယ်ရီ ပိုင်)
*   **သဘောတရား:** အကြွေးဝယ်ကတ် (Credit Card) အရွယ်အစားပဲရှိတဲ့ ကွန်ပျူတာ အသေးစားလေးပါ။ သူ့မှာ GPU မပါပါဘူး။ ARM CPU ဖြင့်သာ အလုပ်လုပ်ပါတယ်။
*   **အသုံးပြုမှု:** ပေါ့ပါးတဲ့ သာမန် Classification Models တွေ၊ မျက်နှာရှာတဲ့ (Haar Cascade / RetinaFace) တွေအတွက် အဆင်ပြေပါတယ်။ YOLOv8 ကို ONNX ပြောင်းပြီး သုံးရင်တောင် FPS (Frame Per Second) 1 ကနေ 5 လောက်ပဲ ထွက်နိုင်ပါတယ်။ သိပ်မလိုအပ်တဲ့ (Low-power) နေရာတွေမှာ သုံးပါတယ်။

### (B) NVIDIA Jetson Family (Nano, Orin Nano) 🔥
*   **သဘောတရား:** NVIDIA ကနေ AI အတွက် သီးသန့် ထုတ်ထားတဲ့ "Mini Supercomputer" လေးတွေပါ။ သူ့ရဲ့ အကြီးမားဆုံး အားသာချက်က **GPU ပါဝင်ပြီး CUDA နဲ့ TensorRT ကို အပြည့်အဝ အသုံးပြုနိုင်ခြင်း** ဖြစ်ပါတယ်။
*   **အသုံးပြုမှု:** Computer Vision သမားတွေအတွက် အချစ်တော် စက်လေးတွေပါ။ YOLOv8 လို Model တွေကို TensorRT (INT8 Quantization) ပြောင်းပြီး Run လိုက်ရင် **30 FPS ကနေ 60 FPS အထိ** အလွန်ချောမွေ့စွာ Real-time Detection လုပ်နိုင်ပါတယ်။ စက်ရုပ်တွေ၊ ဒရုန်းတွေမှာ အများဆုံး သုံးပါတယ်။

### (C) Google Coral (Edge TPU)
*   **သဘောတရား:** USB Flash Drive အရွယ်အစားလေးဖြစ်ပြီး، Raspberry Pi လို စက်တွေမှာ ထိုးစိုက်လိုက်တာနဲ့ AI တွက်ချက်မှု အဆ ၅၀ လောက် မြန်သွားစေတဲ့ အရာပါ။ Google ရဲ့ TPU (Tensor Processing Unit) ကို အသုံးပြုထားပါတယ်။
*   **အသုံးပြုမှု:** Model ကို TensorFlow Lite (TFLite) Format သို့ ပြောင်းလဲထားရန် မဖြစ်မနေ လိုအပ်ပါသည်။

---

## 🛠️ ၃။ Edge Device ပေါ်တွင် Deploy လုပ်မည့် Workflow

ကွန်ပျူတာအကြီးကြီးပေါ်က Model ကို စက်အသေးလေးပေါ် ရွှေ့တဲ့ လုပ်ငန်းစဉ်ပါ။

1.  **Train on Cloud:** Data အများကြီးနဲ့ Model ကို Google Colab သို့မဟုတ် PC ပေါ်မှာ PyTorch ဖြင့် အရင် Train ပါ။
2.  **Optimize & Export:** Chapter 10.1 တွင် သင်ခဲ့သည့်အတိုင်း Model ကို `ONNX` သို့မဟုတ် `TensorRT (.engine)` format သို့ Export လုပ်ပါ။ (မှတ်ချက် - INT8 သို့ Quantization လုပ်ပါက Size အလွန်သေးသွားပြီး ပိုမြန်မည်)။
3.  **Transfer to Edge:** ရလာတဲ့ ဖိုင်လေး (ဥပမာ 15MB ခန့်) ကို Raspberry Pi သို့မဟုတ် Jetson ထဲသို့ (USB သို့မဟုတ် Wi-Fi မှတစ်ဆင့်) ကူးထည့်ပါ။
4.  **Run Inference:** Edge Device ပေါ်တွင် PyTorch ကြီးတစ်ခုလုံးကို Install လုပ်စရာမလိုပါ။ ပေါ့ပါးသော `onnxruntime` ဖြင့်သာ Code ရေး၍ Run ပါမည်။

---

## 💻 ၄။ Raspberry Pi တွင် ONNX Runtime ဖြင့် အလွယ်ဆုံး Run ကြည့်ခြင်း

PyTorch က Library အရွယ်အစား 2GB ကျော်ရှိပါတယ်။ အဲ့ဒါကို Raspberry Pi ပေါ်မှာ သွားသွင်းရင် အရမ်းလေးပါတယ်။ ဒါကြောင့် C++ နဲ့ ရေးထားပြီး အလွန်မြန်ဆန်ပေါ့ပါးတဲ့ `onnxruntime` ကို သုံးလေ့ရှိပါတယ်။

**Installation (on Edge Device):**
```bash
pip install onnxruntime opencv-python
```

**Inference Code (Python):**
```python
import cv2
import numpy as np
import onnxruntime as ort

def run_on_edge():
    # ၁။ ONNX Runtime Session ကို စတင်ခြင်း (PyTorch မလိုတော့ပါ!)
    # CPU ဖြင့်သာ Run မည်ဖြစ်သောကြောင့် 'CPUExecutionProvider' ကို သုံးသည်
    session = ort.InferenceSession("yolov8n.onnx", providers=['CPUExecutionProvider'])
    
    # Model ၏ Input အမည်ကို ယူခြင်း (များသောအားဖြင့် 'images')
    input_name = session.get_inputs()[0].name

    # ၂။ ကင်မရာ (Webcam) မှ Video ဖမ်းယူခြင်း
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # ၃။ Image Preprocessing (ONNX Model နားလည်အောင် ပြင်ဆင်ခြင်း)
        # BGR ကို RGB ပြောင်းသည်၊ 640x640 သို့ အရွယ်အစားညှိသည်
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        
        # HWC မှ CHW သို့ပြောင်းသည်၊ 0-1 Normalization လုပ်သည်
        img_data = img.transpose((2, 0, 1)) / 255.0 
        img_data = np.expand_dims(img_data, axis=0).astype(np.float32)

        # ၄။ Inference (ခန့်မှန်းခြင်း) ပြုလုပ်ခြင်း
        # session.run သည် PyTorch ၏ forward pass ထက် အများကြီး ပိုမြန်ပါသည်
        results = session.run(None, {input_name: img_data})

        # ရလဒ်များကို မျက်နှာပြင်တွင် ပြသခြင်း (Post-processing code များ ဤနေရာတွင် ရေးရန်)
        cv2.imshow("Edge AI Detection", frame)

        # 'q' နှိပ်ပါက ထွက်မည်
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_on_edge()
```
*(ဒီ Code လေးဟာ Raspberry Pi လို CPU သက်သက်စက်တွေပေါ်မှာတောင် PyTorch ထက် အနည်းဆုံး ၂ ဆကနေ ၃ ဆလောက် ပိုမြန်မြန် အလုပ်လုပ်ပေးနိုင်ပါတယ်။)*

---

## 🌡️ ၅။ Edge ပေါ်တွင် အလုပ်လုပ်ရာ၌ သတိပြုရမည့် အချက်များ (Best Practices)

1.  **Thermal Throttling (အပူလွန်ကဲခြင်း):** Edge Device တွေက AI ကို ၂၄ နာရီ Run တဲ့အခါ အလွန်ပူလာပါတယ်။ ပူလာရင် စက်က မလောင်အောင် CPU အလုပ်လုပ်နှုန်းကို အလိုလို လျှော့ချပစ်ပါတယ်။ ဒါကြောင့် **Heat Sink နဲ့ Cooling Fan တွေ မဖြစ်မနေ တပ်ဆင်ထားဖို့ လိုပါတယ်။**
2.  **Frame Skipping:** FPS မနိုင်ဘူးဆိုရင် Video ရဲ့ Frame တိုင်းကို စစ်စရာ မလိုပါဘူး။ "၃ ပုံမှာ ၁ ပုံကိုပဲ (Skip လုပ်ပြီး) AI ကို စစ်ခိုင်းမယ်၊ ကျန်တဲ့အချိန်မှာ Tracking နဲ့ပဲ လိုက်မယ်" ဆိုတဲ့ Logic မျိုး ရေးသင့်ပါတယ်။

---

## 📝 အနှစ်ချုပ် (Key Takeaways)
1.  **Edge AI** သည် Data များကို Cloud သို့ မပို့ဘဲ Hardware အသေးစားလေးများပေါ်တွင် တိုက်ရိုက်တွက်ချက်သော စနစ်ဖြစ်သည်။ Latency, Privacy နှင့် Bandwidth ပြဿနာများကို ဖြေရှင်းပေးသည်။
2.  **Raspberry Pi** သည် CPU သက်သက်ဖြစ်၍ ပေါ့ပါးသော Model များသာ အဆင်ပြေပြီး၊ **NVIDIA Jetson** သည် GPU နှင့် TensorRT ပါဝင်သဖြင့် Heavy Model (YOLO) များအတွက် အကောင်းဆုံး ဖြစ်သည်။
3.  Edge ပေါ်တွင် PyTorch အစား **ONNX Runtime** သို့မဟုတ် **TensorRT** ကို အသုံးပြုခြင်းဖြင့် Memory လွတ်ကင်းပြီး အမြန်ဆုံး အလုပ်လုပ်နိုင်သည်။

> **🔜 Next Step:**
> Hardware ပေါ်မှာ Deploy လုပ်တာကို သိပြီဆိုတော့ Software (Web) ပေါ်မှာရော ဘယ်လို Deploy လုပ်မလဲ? 
> "ကိုယ်ရေးထားတဲ့ Computer Vision Model ကြီးကို မိုဘိုင်း အက်ပ် (Mobile App) တွေ၊ Web တွေကနေ လှမ်းသုံးလို့ရအောင် အင်တာနက်ပေါ်က တံခါးပေါက် (API) တွေ ဘယ်လိုဆောက်မလဲ?"
> ရှေ့ဆက်မယ့် **Chapter 11.1: Building Computer Vision APIs (FastAPI)** မှာ Backend Developer လိုမျိုး AI API လေးတစ်ခုကို လက်တွေ့ တည်ဆောက်သွားကြပါမယ်။

---
*If you find this useful, don't forget to ⭐ star the repository!*

---
