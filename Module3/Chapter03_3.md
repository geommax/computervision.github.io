


GitHub Knowledge Sharing Page အတွက် **Module 3: Chapter 3.3** ကို ဆက်လက် ရေးသားပေးလိုက်ပါတယ်။ ဒီအခန်းဟာ Deep Learning တွေ မပေါ်ခင်ခေတ်ကတည်းက Object Detection (အရာဝတ္ထုရှာဖွေခြင်း) အတွက် အဓိက အသုံးပြုခဲ့ကြပြီး၊ ယနေ့ခေတ် Document Scanner (စာရွက်စကင်ဖတ်သည့်စနစ်) တွေ၊ ကားလမ်းကြောင်းရှာဖွေတဲ့ စနစ် (Lane Detection) တွေမှာ အဓိက သုံးနေဆဲဖြစ်တဲ့ အလွန်အရေးပါတဲ့ အခန်းဖြစ်ပါတယ်။

---

# 📘 Chapter 3.3: Edge & Contour Detection (အနားသတ်မျဉ်းများနှင့် ပုံသဏ္ဌာန်များ ခြေရာခံခြင်း)

လူသားတွေဟာ အရာဝတ္ထုတစ်ခုကို ကြည့်တဲ့အခါ အရောင်တွေ မပါဘဲ မျဉ်းကြောင်း (Outline) သက်သက် ရေးဆွဲထားရင်တောင် အဲ့ဒါ ဘာလဲဆိုတာကို သိနိုင်ပါတယ်။ ကွန်ပျူတာတွေအတွက်လည်း အရာဝတ္ထုတွေရဲ့ အနားသတ်မျဉ်း (Edges) တွေကို ရှာဖွေနိုင်ခြင်းဟာ အလွန်အရေးပါပါတယ်။

ဒီအခန်းမှာ ပုံထဲက အရာဝတ္ထုတွေရဲ့ အနားသတ်တွေကို **Canny Edge Detection** နဲ့ ရှာဖွေပြီး၊ အဲ့ဒီအနားသတ်တွေကို ဆက်စပ်ကာ ပုံသဏ္ဌာန်တစ်ခုလုံးကို ခြေရာခံမယ့် **Contours** သဘောတရားတွေကို လေ့လာသွားပါမယ်။

---

## 🔪 ၁။ Canny Edge Detection (အနားသတ်မျဉ်းများ ရှာဖွေခြင်း)

OpenCV မှာ Edge ရှာတဲ့ နည်းလမ်းများစွာ (Sobel, Laplacian) ရှိပေမယ့် အကောင်းဆုံးနဲ့ အသုံးအများဆုံးကတော့ **Canny Edge Detector** ပဲ ဖြစ်ပါတယ်။ 

**Canny အလုပ်လုပ်ပုံ:** သူက ပုံထဲမှာရှိတဲ့ အရောင်တန်ဖိုး (Pixel Intensity) ရုတ်တရက် ပြောင်းလဲသွားတဲ့ နေရာတွေကို လိုက်ရှာတာ ဖြစ်ပါတယ်။ (ဥပမာ - အဖြူရောင် နောက်ခံပေါ်မှာ အမည်းရောင် စာလုံးလေး ရှိနေရင်၊ အဖြူနဲ့ အမည်း ထိတွေ့နေတဲ့ မျဉ်းကြောင်းကို အနားသတ်အဖြစ် သတ်မှတ်ပါတယ်)။

> ⚠️ **အရေးကြီးသော အချက်:** Canny Edge ကို မသုံးခင်မှာ ပုံကို **Grayscale (အဖြူအမည်း)** ပြောင်းပြီး **Gaussian Blur (ဝါးခြင်း)** အရင်လုပ်ပေးဖို့ မဖြစ်မနေ လိုအပ်ပါတယ်။ (Noise တွေကြောင့် အနားသတ်မျဉ်း အမှားတွေ ထွက်မလာအောင် ဖြစ်ပါတယ်)။

```python
import cv2

image = cv2.imread('test.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny Edge Detection ပြုလုပ်ခြင်း
# Threshold အနိမ့် (50) နှင့် အမြင့် (150) ကို သတ်မှတ်ပေးရသည်
edges = cv2.Canny(blurred, 50, 150)

cv2.imshow('Edges', edges)
```
*(Threshold တန်ဖိုးတွေကို ပုံအပေါ်မူတည်ပြီး ကစားပေးရပါတယ်။ တန်ဖိုးနည်းရင် အနားသတ်တွေ အများကြီး ထွက်လာပြီး၊ တန်ဖိုးများရင်တော့ အထင်ရှားဆုံး အနားသတ်တွေကိုပဲ တွေ့ရပါမယ်။)*

---

## 🔗 ၂။ Find Contours (ပုံသဏ္ဌာန်များ ခြေရာခံခြင်း)

Canny ကနေ ထွက်လာတဲ့ Edges တွေဟာ ကွန်ပျူတာအတွက်တော့ အဖြူရောင် အစက်လေးတွေ (Pixels) စုနေတာ သက်သက်ပါပဲ။ အဲ့ဒီ အဖြူရောင် မျဉ်းကြောင်းလေးတွေဟာ ဘယ်နေရာကစပြီး ဘယ်နေရာမှာ ဆုံးတယ်၊ တစ်ဆက်တည်း ဖြစ်နေတဲ့ ပုံသဏ္ဌာန် (Object) တစ်ခုလား ဆိုတာကို ကွန်ပျူတာ နားလည်အောင် **Contours** ရှာပေးရပါတယ်။

**Contour ဆိုတာ တူညီတဲ့ အရောင်ရှိတဲ့ အဆက်မပြတ် မျဉ်းကြောင်းတွေကို တွဲဆက်ပေးလိုက်ခြင်း** ဖြစ်ပါတယ်။

```python
# edges ဆိုသည်မှာ Canny မှ ထွက်လာသော အဖြူအမည်းပုံ ဖြစ်ရမည်
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

**Parameters ရှင်းလင်းချက်:**
1.  **`cv2.RETR_EXTERNAL` (Retrieval Mode):** အပြင်ဘက်ဆုံးက ပုံသဏ္ဌာန်တွေကိုပဲ ရှာမယ်လို့ ဆိုလိုတာပါ။ (အကယ်၍ အရာဝတ္ထုတစ်ခုရဲ့ အတွင်းထဲက အပေါက်လေးတွေကိုပါ ရှာချင်ရင် `cv2.RETR_TREE` ကို သုံးရပါတယ်)။
2.  **`cv2.CHAIN_APPROX_SIMPLE` (Approximation Method):** မှတ်ဉာဏ်သက်သာအောင် ပြုလုပ်ခြင်းပါ။ လေးထောင့်ပုံတစ်ခုကို Contour ရှာရင် မျဉ်းကြောင်းပေါ်က Pixel တွေအားလုံးကို မမှတ်တော့ဘဲ၊ ထောင့် ၄ ထောင့်ရဲ့ (x,y) ကိုပဲ မှတ်ထားပေးတာ ဖြစ်ပါတယ်။

---

## 🎨 ၃။ Draw Contours & Bounding Boxes (ခြေရာခံမိသည်များကို ရေးဆွဲခြင်း)

Contours တွေကို ရှာတွေ့သွားပြီဆိုရင် မူရင်းပုံပေါ်မှာ ပြန်ပြီး ရေးဆွဲပြသနိုင်ပါတယ်။ ရေးဆွဲနည်း ၂ မျိုး ရှိပါတယ်။

### (A) Contour မျဉ်းကြောင်းအတိုင်း အတိအကျဆွဲခြင်း
```python
# -1 ဆိုသည်မှာ တွေ့သမျှ Contour အားလုံးကို ဆွဲမည်ဟု ဆိုလိုသည်
# (0, 255, 0) သည် အစိမ်းရောင်၊ 2 သည် မျဉ်းအထူ
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
```

### (B) Bounding Box (စတုဂံဘောင်) ဖြင့် ဝိုင်းပြခြင်း
လက်တွေ့မှာ မျဉ်းကြောင်းအတိုင်း ကွေ့ကောက်ပြီး ဆွဲတာထက်၊ အရာဝတ္ထုတစ်ခုလုံးကို လေးထောင့်ဘောင် ခတ်ပြတာ (Object Detection Style) ကို ပိုပြီး အသုံးပြုကြပါတယ်။

```python
# တွေ့သမျှ Contour တစ်ခုချင်းစီကို Loop ပတ်မည်
for contour in contours:
    # Contour တစ်ခု၏ အသေးဆုံး လေးထောင့်ဘောင် (x, y, width, height) ကို တွက်ထုတ်မည်
    x, y, w, h = cv2.boundingRect(contour)
    
    # ထို (x,y) မှစ၍ စတုဂံဘောင် ဆွဲမည်
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
```

---

## 🧩 ၄။ အစအဆုံး ပေါင်းရေးကြည့်ခြင်း (Complete Example: Object Outliner)

အခု သင်ခဲ့တာတွေကို ပေါင်းပြီး ပုံထဲက အရာဝတ္ထုတွေကို အလိုအလျောက် ရှာဖွေပြီး လေးထောင့်ဘောင် ခတ်ပေးမယ့် Script လေးတစ်ခု ရေးကြည့်ကြရအောင်။

```python
import cv2

def main():
    # ၁။ မူရင်းပုံကို ဖတ်ခြင်း
    image = cv2.imread('objects.jpg') # ပစ္စည်းများစွာ ပါဝင်သော ပုံဖြစ်သင့်သည်
    if image is None:
        print("Image not found!")
        return
        
    image = cv2.resize(image, (800, 600))
    original_copy = image.copy() # Contour ဆွဲရန် Copy ကူးထားမည်

    # ၂။ Preprocessing လုပ်ခြင်း (Gray -> Blur)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ၃။ Canny Edge ရှာဖွေခြင်း
    edges = cv2.Canny(blurred, 50, 150)

    # ၄။ Contours ခြေရာခံခြင်း
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Total objects found: {len(contours)}")

    # ၅။ တွေ့ရှိသော Contours များကို Bounding Box ဆွဲခြင်း
    for contour in contours:
        # အရမ်းသေးငယ်သော Noise များကို ဖယ်ရှားရန် Area စစ်ဆေးခြင်း
        if cv2.contourArea(contour) > 500: 
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(original_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # စာသားလေးပါ ထည့်မည်
            cv2.putText(original_copy, "Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ရလဒ်များ ပြသခြင်း
    cv2.imshow('1. Canny Edges', edges)
    cv2.imshow('2. Detected Objects', original_copy)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```
*(ရှင်းလင်းချက်: `cv2.contourArea(contour)` ဟာ အလွန်အသုံးဝင်ပါတယ်။ ပုံထဲမှာရှိတဲ့ အစက်အပြောက် သေးသေးလေးတွေကို Contour အဖြစ် မသတ်မှတ်မိအောင် ဧရိယာ 500 ထက်ကြီးတဲ့ အရာဝတ္ထုတွေကိုပဲ ဘောင်ခတ်ပြအောင် ထိန်းချုပ်ထားခြင်း ဖြစ်ပါတယ်။)*

---

## 📝 အနှစ်ချုပ် (Key Takeaways)
1.  **Canny Edge Detection** မလုပ်မီ ပုံကို Grayscale နှင့် Blur အမြဲတမ်း အရင်လုပ်ပေးရပါမည်။
2.  **`cv2.Canny(image, minVal, maxVal)`** ဖြင့် အရာဝတ္ထုများ၏ အနားသတ်မျဉ်းများကို ရှာဖွေနိုင်သည်။
3.  **`cv2.findContours()`** သည် Canny မှ ထွက်လာသော မျဉ်းကြောင်းများကို အဆက်အစပ်ရှိသော ပုံသဏ္ဌာန် (Object) တစ်ခုအဖြစ် ပေါင်းစည်း ခြေရာခံပေးသည်။
4.  **`cv2.boundingRect()`** ကို အသုံးပြု၍ Contour ၏ (x, y, width, height) ကို တွက်ထုတ်ပြီး Object Detection များကဲ့သို့ စတုဂံဘောင်များ ရေးဆွဲနိုင်သည်။

> **🔜 Next Step:**
> ရှေ့ဆက်မယ့် **Chapter 3.4: Traditional Feature Extraction (Overview)** မှာတော့ Deep Learning တွေ မပေါ်ခင်က မျက်နှာတွေကို ဘယ်လိုရှာလဲ (Haar Cascades)၊ ပုံ ၂ ပုံ တူ/မတူ ဘယ်လိုတိုက်စစ်လဲ (SIFT/ORB) ဆိုတဲ့ သဘောတရားတွေကို ဗဟုသုတအနေနဲ့ အကျဉ်းချုပ် လေ့လာပြီးရင် **Part 1** တစ်ခုလုံး အောင်မြင်စွာ ပြီးဆုံးတော့မှာ ဖြစ်ပါတယ်။

---
*If you find this useful, don't forget to ⭐ star the repository!*

---
