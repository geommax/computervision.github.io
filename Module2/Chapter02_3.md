


GitHub Knowledge Sharing Page အတွက် **Module 2: Chapter 2.3** ကို ဆက်လက် ရေးသားပေးလိုက်ပါတယ်။ ဒီအခန်းဟာ နောက်ပိုင်း Object Detection (ဥပမာ - YOLO) တွေ လုပ်တဲ့အခါ AI က ရှာတွေ့တဲ့ အရာဝတ္ထုတွေကို လေးထောင့်ဘောင် (Bounding Box) တွေ ဆွဲတာ၊ စာသား (Label) တွေ တပ်တာတွေမှာ မရှိမဖြစ် အသုံးပြုရမယ့် အခန်းဖြစ်ပါတယ်။

---

# 📘 Chapter 2.3: Image Manipulation & Drawing (Cropping, Resizing နှင့် ပုံဆွဲခြင်းများ)

Computer Vision Model (ဥပမာ - CNN သို့မဟုတ် YOLO) တွေဟာ ပုံမှန်အားဖြင့် အရွယ်အစားအကြီးကြီး (ဥပမာ 4K Resolution) တွေကို တိုက်ရိုက် လက်မခံနိုင်ပါဘူး။ သူတို့ သတ်မှတ်ထားတဲ့ အရွယ်အစား (ဥပမာ `224x224` သို့မဟုတ် `640x640`) ကို ပြောင်းလဲပေးရလေ့ရှိပါတယ်။ ဒါ့အပြင် AI က ရှာတွေ့တဲ့ အရာဝတ္ထုတွေကို လူတွေ မြင်သာအောင် ဘောင်တွေ၊ စာသားတွေနဲ့ ပြသပေးဖို့လည်း လိုအပ်ပါတယ်။

ဒီအခန်းမှာ ပုံတွေကို အရွယ်အစားပြောင်းခြင်း (Resizing)၊ လိုချင်တဲ့ အပိုင်းကို ဖြတ်ယူခြင်း (Cropping) နဲ့ ပုံပေါ်မှာ ပုံသဏ္ဌာန်တွေ ရေးဆွဲခြင်း (Drawing) တွေကို လေ့လာသွားပါမယ်။

---

## 📏 ၁။ ပုံများကို အရွယ်အစားပြောင်းခြင်း (`cv2.resize`)

ပုံတစ်ပုံကို အရွယ်အစား သေးချင်/ကြီးချင်တဲ့အခါ `cv2.resize()` ကို အသုံးပြုပါတယ်။

```python
import cv2

image = cv2.imread('test.jpg')

# Resizing ပြုလုပ်ခြင်း 
# (မှတ်ချက် - ဤနေရာတွင် (Width, Height) အစီအစဉ်ဖြင့် ထည့်ရပါမည်)
resized_image = cv2.resize(image, (640, 480))

cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

> 💡 **Beginner Pitfall (အမှားများလေ့ရှိသော အချက်):**
> အရင်အခန်းတုန်းက `image.shape` ကို စစ်ကြည့်ရင် `(Height, Width)` အစီအစဉ်နဲ့ ထွက်လာတာကို မှတ်မိမှာပါ။ ဒါပေမယ့် `cv2.resize()` မှာ ပေးရမယ့် အရွယ်အစားကတော့ သင်္ချာ x,y ဝင်ရိုးအတိုင်း **`(Width, Height)`** ပြောင်းပြန် ပြန်ဖြစ်သွားပါတယ်။ ဒါကို မကြာခဏ မှားယွင်းတတ်လို့ သတိပြုရပါမယ်။

---

## ✂️ ၂။ လိုချင်သော အစိတ်အပိုင်းကို ဖြတ်ယူခြင်း (Cropping)

OpenCV မှာ ပုံကိုဖြတ်ဖို့ (Crop လုပ်ဖို့) သီးသန့် Function မရှိပါဘူး။ ပုံဆိုတာ Numpy Array (ဇယားကွက်) ကြီး ဖြစ်တဲ့အတွက်၊ Array ရဲ့ လိုချင်တဲ့ အပိုင်းကို ဖြတ်ထုတ်လိုက်ရုံ (Numpy Slicing လုပ်ရုံ) ပါပဲ။

ဖြတ်ထုတ်မယ့် ဖော်မြူလာကတော့ `image[startY:endY, startX:endX]` ဖြစ်ပါတယ်။ 
*(ဒီနေရာမှာ Y (အထက်/အောက်) က အရင်လာပြီး၊ X (ဘယ်/ညာ) က နောက်မှ လာတာကို သတိပြုပါ။)*

```python
import cv2

image = cv2.imread('test.jpg')

# ဥပမာ - Y ဝင်ရိုး 100 မှ 400 အထိ၊ X ဝင်ရိုး 200 မှ 500 အထိကို ဖြတ်ယူမည်
# image[y1:y2, x1:x2]
cropped_image = image[100:400, 200:500]

cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 🔄 ၃။ ပုံများကို လှည့်ခြင်းနှင့် ပြောင်းပြန်လှန်ခြင်း (Flipping)

ပုံတစ်ပုံကို ဘယ်/ညာ ပြောင်းပြန် (Mirror Effect) လှန်ချင်တာပဲဖြစ်ဖြစ်၊ အထက်/အောက် ပြောင်းချင်တာပဲဖြစ်ဖြစ် `cv2.flip()` ကို အသုံးပြုနိုင်ပါတယ်။

```python
import cv2

image = cv2.imread('test.jpg')

# flipCode = 1 (ဘယ်/ညာ ပြောင်းပြန်လှန်ခြင်း - Horizontal Flip)
flipped_horizontal = cv2.flip(image, 1)

# flipCode = 0 (အထက်/အောက် ပြောင်းပြန်လှန်ခြင်း - Vertical Flip)
flipped_vertical = cv2.flip(image, 0)

# flipCode = -1 (ဘယ်/ညာ ရော၊ အထက်/အောက် ပါ ပြောင်းပြန်လှန်ခြင်း)
flipped_both = cv2.flip(image, -1)
```
*(Flipping ကို နောက်ပိုင်း Data Augmentation (ပုံပွားခြင်း) အခန်းတွေမှာ အများဆုံး အသုံးပြုပါမယ်။)*

---

## 🎨 ၄။ ပုံပေါ်တွင် ရေးဆွဲခြင်း (Drawing Functions)

Computer Vision မှာ အများဆုံး သုံးရမယ့် Drawing Function ၃ ခု ရှိပါတယ်။ အားလုံးမှာ တူညီတဲ့ စည်းမျဉ်းတစ်ခုကတော့ **အရောင်ကို `(Blue, Green, Red)`** အစီအစဉ်အတိုင်း ထည့်ရခြင်းပဲ ဖြစ်ပါတယ်။ (ဥပမာ - အစိမ်းရောင် ဆိုလျှင် `(0, 255, 0)` ဖြစ်ပါသည်)။

### (A) စတုဂံဘောင် ဆွဲခြင်း (Rectangle for Bounding Box)
Object Detection မှာ အရာဝတ္ထုတွေကို ဝိုင်းပြဖို့ အသုံးပြုပါတယ်။ ဘောင်ဆွဲဖို့ဆိုရင် ဘယ်ဘက်-အပေါ်ထောင့် `(x1, y1)` နဲ့ ညာဘက်-အောက်ထောင့် `(x2, y2)` ကို သိဖို့လိုပါတယ်။

```python
# cv2.rectangle(image, top_left_point, bottom_right_point, color, thickness)
cv2.rectangle(image, (200, 100), (500, 400), (0, 255, 0), 3) 
# အစိမ်းရောင် (0, 255, 0) ဖြင့် အထူ 3px ရှိသော ဘောင်ဆွဲမည်
```

### (B) စက်ဝိုင်း ဆွဲခြင်း (Circle)
မျက်နှာပြင်ပေါ်က အမှတ်အသားတွေ (ဥပမာ - မျက်လုံး၊ နှာခေါင်း နေရာ) ကို ပြသဖို့ အသုံးများပါတယ်။ စက်ဝိုင်းဗဟိုချက် `(x, y)` နဲ့ အချင်းဝက် (Radius) လိုအပ်ပါတယ်။

```python
# cv2.circle(image, center_point, radius, color, thickness)
cv2.circle(image, (350, 250), 50, (0, 0, 255), -1) 
# အနီရောင် (0, 0, 255) ဖြင့် ဆွဲမည်။ thickness = -1 ဆိုလျှင် စက်ဝိုင်းအပြည့် ခြယ်မည်။
```

### (C) စာသား ရေးထည့်ခြင်း (Putting Text for Labels)
ရှာတွေ့တဲ့ အရာဝတ္ထုရဲ့ နာမည် (ဥပမာ - "Dog 95%") ကို ဘောင်ရဲ့ အပေါ်မှာ ရေးပြဖို့ အသုံးပြုပါတယ်။

```python
# cv2.putText(image, text, origin_point, font, font_scale, color, thickness)
cv2.putText(image, 'Object Detected', (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
# အပြာရောင် (255, 0, 0) ဖြင့် ရေးမည်။
```

---

## 🧩 ၅။ အစအဆုံး ပေါင်းရေးကြည့်ခြင်း (Complete Example: AI Bounding Box Simulator)

အပေါ်မှာ သင်ခဲ့တာတွေကို ပေါင်းပြီး AI က အရာဝတ္ထုတစ်ခုကို ရှာတွေ့သွားတဲ့အခါ မျက်နှာပြင်ပေါ်မှာ ဘယ်လိုပြသမလဲဆိုတာကို သရုပ်ဖော်ကြည့်ရအောင်။

```python
import cv2

def main():
    # ၁။ ပုံကို ဖတ်ခြင်း
    image = cv2.imread('test.jpg')
    if image is None:
        print("Image not found!")
        return

    # ၂။ အရင်ဆုံး ပုံကို လုပ်ရကိုင်ရ လွယ်အောင် Resize လုပ်မည်
    # (Width=800, Height=600)
    image = cv2.resize(image, (800, 600))

    # ၃။ AI က ရှာတွေ့သည်ဟု ယူဆသော နေရာကို Bounding Box စတုဂံဆွဲမည်
    # (x1, y1) = (300, 150), (x2, y2) = (550, 500)
    top_left = (300, 150)
    bottom_right = (550, 500)
    box_color = (0, 255, 0) # Green
    cv2.rectangle(image, top_left, bottom_right, box_color, 3)

    # ၄။ ဘောင်၏ အပေါ်တွင် Label စာသား ရေးထည့်မည်
    text = "Person 98%"
    text_position = (300, 140) # ဘောင်၏ အပေါ်နား
    text_color = (0, 255, 0)
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    # ၅။ အရာဝတ္ထု၏ အလယ်ဗဟိုကို အမှတ်အသားပြုရန် စက်ဝိုင်းအနီ ဆွဲမည်
    center_point = (425, 325)
    cv2.circle(image, center_point, 5, (0, 0, 255), -1)

    # ၆။ မျက်နှာပြင်တွင် ပြသခြင်း
    cv2.imshow('AI Detection Simulator', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

*(သင့်စက်ထဲက ပုံတစ်ပုံကို `test.jpg` အဖြစ်ထားပြီး အထက်ပါ Code ကို Run ကြည့်ပါ။ AI က ထောက်လှမ်းထားသလိုမျိုး လှပတဲ့ Bounding Box လေး ထွက်လာတာကို မြင်တွေ့ရပါလိမ့်မယ်။)*

---

## 📝 အနှစ်ချုပ် (Key Takeaways)
1.  **`cv2.resize()`** ကို သုံးရာတွင် `(Width, Height)` အစီအစဉ် ဖြစ်ကြောင်း သတိပြုပါ။
2.  **Cropping** လုပ်ရန် သီးသန့် Function မလိုဘဲ Numpy Slicing `image[y1:y2, x1:x2]` ကို သုံးရသည်။ (Y က အရင်လာသည်)။
3.  **Drawing Functions** (Rectangle, Circle, Text) များကို အသုံးပြု၍ AI ၏ ရလဒ်များကို လူသားများ မြင်သာအောင် ပြသနိုင်သည်။
4.  OpenCV တွင် အရောင်များကို **`BGR` (Blue, Green, Red)** အစီအစဉ်ဖြင့် အမြဲရေးရသည်။

> **🔜 Next Step:**
> ရှေ့ဆက်မယ့် **Chapter 2.4: Color Spaces & Masking** မှာတော့ အရောင်တွေရဲ့ အလုပ်လုပ်ပုံကို ပိုမိုနက်ရှိုင်းစွာ လေ့လာပြီး၊ ပုံထဲကနေ ကိုယ်လိုချင်တဲ့ အရောင် (ဥပမာ - အစိမ်းရောင် အရာဝတ္ထု) ကို သီးသန့်ခွဲထုတ် (Color Masking) တဲ့ နည်းပညာတွေကို လေ့လာသွားကြပါမယ်။

---
*If you find this useful, don't forget to ⭐ star the repository!*

---
