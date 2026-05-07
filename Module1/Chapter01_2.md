


GitHub Knowledge Sharing Page အတွက် **Chapter 1.2** ကို တောင်းဆိုထားတဲ့အတိုင်း တခြားအကြောင်းအရာတွေ မပါဘဲ **Python Virtual Environment (`venv`)** တည်ဆောက်တဲ့ အပိုင်းသီးသန့်ကို အသေးစိတ် ရေးသားပေးလိုက်ပါတယ်။

---

# 📘 Chapter 1.2: Setting Up the AI Workspace (Python Virtual Environment တည်ဆောက်ခြင်း)

Computer Vision Project တွေ၊ AI Project တွေ ရေးသားတဲ့အခါ Project တစ်ခုနဲ့ တစ်ခု အသုံးပြုတဲ့ Library Version တွေ (ဥပမာ - OpenCV, PyTorch) မတူညီတာမျိုး ကြုံရတတ်ပါတယ်။ Project A က PyTorch အဟောင်းကို သုံးထားပြီး၊ Project B က အသစ်ကို သုံးချင်တာမျိုး ဖြစ်နိုင်ပါတယ်။

ဒီလိုအချိန်မှာ Python Library တွေ အချင်းချင်း ရောထွေးပြီး Error မတက်အောင် Project တစ်ခုချင်းစီအတွက် **သီးသန့် အခန်းငယ်လေး (Isolated Environment)** တစ်ခုစီ ဖန်တီးပေးဖို့ လိုအပ်ပါတယ်။ ဒါကို Python မှာ **Virtual Environment (`venv`)** လို့ ခေါ်ပါတယ်။

ဒီအခန်းမှာ စနစ်တကျ `venv` ဆောက်နည်း၊ အသုံးပြုနည်းတွေကို လေ့လာသွားပါမယ်။

---

## 🛠️ အဆင့် (၁) - Python Version စစ်ဆေးခြင်း

အရင်ဆုံး မိမိစက်ထဲမှာ Python သွင်းထားပြီးသား ဟုတ်/မဟုတ် နဲ့ ဘယ် Version သွင်းထားလဲဆိုတာကို စစ်ဆေးပါမယ်။ Terminal (macOS/Linux) သို့မဟုတ် Command Prompt / PowerShell (Windows) ကို ဖွင့်ပြီး အောက်ပါ Command ကို ရိုက်ထည့်ပါ။

```bash
python --version
# သို့မဟုတ်
python3 --version
```

`Python 3.x.x` ဆိုပြီး ပေါ်လာတယ်ဆိုရင် နောက်တစ်ဆင့်ကို ဆက်သွားလို့ ရပါပြီ။

---

## 🏗️ အဆင့် (၂) - Virtual Environment (`venv`) အသစ်တည်ဆောက်ခြင်း

မိမိ Project လုပ်မယ့် Folder ထဲကို Terminal (သို့မဟုတ် CMD) ကနေ ဝင်ပါ။ ထို့နောက် အောက်ပါ Command ကို အသုံးပြုပြီး `venv` ကို တည်ဆောက်နိုင်ပါတယ်။

```bash
python -m venv cv_env
# သို့မဟုတ် macOS/Linux များတွင်
python3 -m venv cv_env
```

> 💡 **မှတ်ချက်:** `cv_env` ဆိုတာက မိမိပေးချင်တဲ့ Virtual Environment နာမည် ဖြစ်ပါတယ်။ `myenv`, `vision_env` စသဖြင့် ကြိုက်နှစ်သက်ရာ နာမည်ပေးနိုင်ပါတယ်။ 

ဒီ Command ကို Run လိုက်ရင် သင်ရောက်နေတဲ့ Folder ထဲမှာ `cv_env` ဆိုတဲ့ Folder အသစ်တစ်ခု အလိုအလျောက် ထွက်လာပါလိမ့်မယ်။ အဲ့ဒီ Folder ထဲမှာ ဒီ Project အတွက် သီးသန့်သုံးမယ့် Python နဲ့ လိုအပ်တဲ့ Libraries တွေ သိမ်းဆည်းပေးသွားမှာ ဖြစ်ပါတယ်။

---

## 🟢 အဆင့် (၃) - Virtual Environment ကို အသက်သွင်းခြင်း (Activating)

Environment ဆောက်ပြီးသွားပေမယ့် ချက်ချင်း အသုံးပြုလို့ မရသေးပါဘူး။ သူ့ကို Active ဖြစ်အောင် (အသက်ဝင်လာအောင်) အရင်လုပ်ပေးရပါမယ်။ မိမိအသုံးပြုတဲ့ OS ပေါ်မူတည်ပြီး Command ကွာခြားပါတယ်။

**Windows (Command Prompt):**
```cmd
cv_env\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.\cv_env\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source cv_env/bin/activate
```

**✅ အောင်မြင်မှု အမှတ်အသား:**
Active ဖြစ်သွားပြီဆိုရင် Terminal ရဲ့ ရှေ့ဆုံးမှာ ကွင်းစကွင်းပိတ်လေးနဲ့ `(cv_env)` ဆိုပြီး ပေါ်လာပါလိမ့်မယ်။ အောက်ပါအတိုင်း မြင်ရပါမယ်။
`(cv_env) C:\Users\MyPC\Computer-Vision-Project>`

ဒီလိုပေါ်နေပြီဆိုရင် သင်သွင်းသမျှ Python Library တွေဟာ သင့်စက်ကြီး တစ်ခုလုံး (Global) ထဲကို မရောက်တော့ဘဲ၊ ဒီ `cv_env` ဆိုတဲ့ သီးသန့်အခန်းလေးထဲကိုသာ ဝင်သွားတော့မှာ ဖြစ်ပါတယ်။ 

---

## 📦 အဆင့် (၄) - Library များ Install လုပ်ကြည့်ခြင်း

Environment Active ဖြစ်နေချိန်မှာ Computer Vision အတွက် မရှိမဖြစ်လိုအပ်တဲ့ OpenCV ကို စမ်းပြီး Install လုပ်ကြည့်ပါမယ်။

```bash
pip install opencv-python
```

Installation ပြီးသွားရင် `pip list` လို့ ရိုက်ထည့်ကြည့်ပါ။ လက်ရှိ Environment ထဲမှာ သွင်းထားတဲ့ Library စာရင်းကို သေသပ်စွာ မြင်တွေ့ရမှာ ဖြစ်ပါတယ်။

---

## 🔴 အဆင့် (၅) - Virtual Environment မှ ထွက်ခြင်း (Deactivating)

မိမိရဲ့ Project ပြီးသွားလို့ဖြစ်စေ၊ အခြား Project တစ်ခုကို ကူးသွားချင်လို့ဖြစ်စေ ဒီ Environment ကနေ ထွက်ချင်တယ်ဆိုရင်တော့ အောက်ပါ Command တစ်ကြောင်းတည်းကို ရိုက်ထည့်လိုက်ရုံပါပဲ။

```bash
deactivate
```

ဒါဆိုရင် Terminal ရှေ့က `(cv_env)` ဆိုတာလေး ပျောက်သွားပြီး ပုံမှန်အခြေအနေကို ပြန်ရောက်သွားပါလိမ့်မယ်။

---

## 📝 အနှစ်ချုပ် (Key Takeaways)
1. `python -m venv <name>` ကိုသုံးပြီး Project တိုင်းအတွက် သီးသန့် Environment တစ်ခု ဆောက်ပါ။
2. အလုပ်မစခင် အမြဲတမ်း `activate` လုပ်ဖို့ မမေ့ပါနဲ့။ (Terminal ရှေ့မှာ နာမည်လေး ပေါ်/မပေါ် စစ်ဆေးပါ)။
3. အလုပ်ပြီးသွားရင် `deactivate` လုပ်ပြီး ထွက်နိုင်ပါတယ်။

> **🔜 Next Step:**
> ရှေ့ဆက်မယ့် **Chapter 1.3** မှာတော့ Computer Vision နဲ့ AI တွေအတွက် အကောင်းဆုံး Framework တစ်ခုဖြစ်တဲ့ **PyTorch** အကြောင်းနဲ့ GPU Acceleration တွေအကြောင်း လေ့လာသွားကြပါမယ်။

--- 
*If you find this useful, don't forget to ⭐ star the repository!*

---
