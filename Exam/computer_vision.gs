function createCVQuiz() {
  // Form အသစ်တည်ဆောက်ခြင်း
  var form = FormApp.create('Computer Vision Complete Exam (Module 1 to 7)');
  
  // Quiz Mode ဖွင့်ခြင်း (အမှတ်နှင့် အဖြေမှန် သတ်မှတ်နိုင်ရန်)
  form.setIsQuiz(true);
  form.setDescription('Computer Vision ဘာသာရပ် Module 1 မှ Module 7 အထိ သင်ခန်းစာများအတွက် Multiple Choice မေးခွန်း ၅၀ ဖြစ်ပါသည်။ မေးခွန်းတစ်ခုလျှင် ၁ မှတ် ဖြစ်ပါသည်။');

  // မေးခွန်း (၅၀) နှင့် အဖြေများ Data Array
  var questionsData =[
    {
      q: "၁။ ကွန်ပျူတာအမြင်အာရုံ (Computer Vision) တွင် Deep Learning မော်ဒယ်များကို Train ရာ၌ CPU ထက် GPU ကို အဘယ်ကြောင့် ပိုမိုအသုံးပြုကြသနည်း?",
      options:["GPU သည် လျှပ်စစ်မီးစားသက်သာသောကြောင့်", "Matrix တွက်ချက်မှု ထောင်ပေါင်းများစွာကို တစ်ပြိုင်နက်တည်း ပြုလုပ်နိုင်သောကြောင့်", "ပုံများ၏ အရောင်ကို ပိုမိုတောက်ပစေသောကြောင့်", "Python သည် GPU တွင်သာ အလုပ်လုပ်သောကြောင့်"],
      ans: 1
    },
    {
      q: "၂။ Python တွင် `cv2.imread()` ဖြင့် ပုံတစ်ပုံကို ဖတ်လိုက်သောအခါ OpenCV သည် မည်သည့် Color Space အစီအစဉ်ဖြင့် ဖတ်ယူသနည်း?",
      options: ["RGB", "Grayscale", "BGR", "HSV"],
      ans: 2
    },
    {
      q: "၃။ OpenCV ၏ Coordinate System အရ ပုံတစ်ပုံ၏ Origin Point (0, 0) သည် မည်သည့်နေရာတွင် ရှိသနည်း?",
      options:["ပုံ၏ အလယ်ဗဟိုတည့်တည့်", "ဘယ်ဘက် အောက်ထောင့် (Bottom-Left)", "ညာဘက် အပေါ်ထောင့် (Top-Right)", "ဘယ်ဘက် အပေါ်ထောင့် (Top-Left)"],
      ans: 3
    },
    {
      q: "၄။ Grayscale (အဖြူအမည်း) ပုံရိပ်တစ်ခုတွင် Color Channel အရေအတွက် မည်မျှပါဝင်သနည်း?",
      options: ["1 Channel", "2 Channels", "3 Channels", "4 Channels"],
      ans: 0
    },
    {
      q: "၅။ OpenCV တွင် အရောင်များကို ရှာဖွေရာ၌ (Color Masking) BGR ထက် မည်သည့် Color Space ကို ပိုမိုအသုံးပြုလေ့ရှိသနည်း?",
      options: ["RGB", "CMYK", "HSV", "LAB"],
      ans: 2
    },
    {
      q: "၆။ OpenCV ၏ `cv2.resize()` ကို အသုံးပြုရာတွင် ပုံ၏ အရွယ်အစားကို မည်သည့်အစီအစဉ်ဖြင့် ထည့်သွင်းရသနည်း?",
      options: ["(Height, Width)", "(Width, Height)", "(Channels, Height, Width)", "(Width, Channels, Height)"],
      ans: 1
    },
    {
      q: "၇။ Python တွင် ကွန်ပျူတာအမြင်အာရုံ ပုံရိပ်များကို မည်သည့် Data Type (ဇယားကွက်) အဖြစ် အဓိက သိမ်းဆည်းသနည်း?",
      options: ["List", "Dictionary", "Pandas DataFrame", "Numpy Array"],
      ans: 3
    },
    {
      q: "၈။ `cv2.waitKey(0)` ၏ အဓိပ္ပာယ်မှာ အဘယ်နည်း?",
      options:["0 စက္ကန့် စောင့်ပြီး Window ကိုပိတ်မည်", "ကီးဘုတ်မှ ခလုတ်တစ်ခုခု နှိပ်သည်အထိ အချိန်အကန့်အသတ်မရှိ စောင့်နေမည်", "Mouse Click ကို စောင့်နေမည်", "Error တက်ပြီး ပရိုဂရမ် ရပ်သွားမည်"],
      ans: 1
    },
    {
      q: "၉။ Artificial Neuron တစ်ခုတွင် Inputs များအားလုံး 0 ဖြစ်နေလျှင်တောင် အခြေခံတန်ဖိုးတစ်ခု (Baseline) ရှိနေစေရန် ပေါင်းထည့်ပေးသော အရာကို မည်သို့ခေါ်သနည်း?",
      options: ["Weight", "Bias", "Activation Function", "Loss"],
      ans: 1
    },
    {
      q: "၁၀။ `image.shape` ကို Print ထုတ်ကြည့်သောအခါ Color ပုံတစ်ပုံအတွက် ထွက်ပေါ်လာမည့် ပုံစံမှာ အဘယ်နည်း?",
      options: ["(Width, Height, Channels)", "(Height, Width, Channels)", "(Channels, Height, Width)", "(Height, Width)"],
      ans: 1
    },
    {
      q: "၁၁။ AI Model တစ်ခု Gradient Descent ဖြင့် အမှားပြင်ဆင်ရာတွင် (Weight Update လုပ်ရာတွင်) ခြေလှမ်းအကျယ်ကို ထိန်းချုပ်ပေးသော ကိန်းဂဏန်းကို မည်သို့ခေါ်သနည်း?",
      options: ["Batch Size", "Learning Rate", "Epoch", "Momentum"],
      ans: 1
    },
    {
      q: "၁၂။ Convolutional Neural Network (CNN) တွင် ပုံရိပ်ပေါ်၌ ရွေ့လျားပြီး အနားသတ်မျဉ်းများနှင့် ပုံသဏ္ဌာန်များကို ရှာဖွေပေးသော ဇယားကွက်ငယ် (Matrix) ကို မည်သို့ခေါ်သနည်း?",
      options: ["Padding", "Kernel (Filter)", "Stride", "Pooling"],
      ans: 1
    },
    {
      q: "၁၃။ Convolution လုပ်ရာတွင် Kernel (Filter) လေးသည် ပုံပေါ်၌ တစ်ကြိမ်လျှင် Pixel မည်မျှခုန်၍ ရွေ့လျားမည်ကို သတ်မှတ်ပေးသော Parameter မှာ အဘယ်နည်း?",
      options: ["Padding", "Stride", "Epoch", "Batch Size"],
      ans: 1
    },
    {
      q: "၁၄။ CNN တွင် Convolution အကြိမ်ကြိမ်လုပ်သောအခါ ပုံ၏အရွယ်အစား သေးငယ်မသွားစေရန်နှင့် အစွန်ဆုံး Pixel များအား တွက်ချက်မှု မလွတ်စေရန် ပုံ၏ဘေးပတ်လည်တွင် 0 များ ထပ်မံကွပ်ပေးခြင်းကို မည်သို့ခေါ်သနည်း?",
      options: ["Flattening", "Pooling", "Padding", "Striding"],
      ans: 2
    },
    {
      q: "၁၅။ Canny Edge Detection ကို အသုံးမပြုမီ ပုံရိပ်အား မည်သည့်လုပ်ငန်းစဉ်များ မဖြစ်မနေ အရင်လုပ်ပေးသင့်သနည်း?",
      options:["Resizing နှင့် Dilation", "Grayscale ပြောင်းခြင်း နှင့် Gaussian Blur လုပ်ခြင်း", "Color Masking လုပ်ခြင်း", "Bounding Box ရေးဆွဲခြင်း"],
      ans: 1
    },
    {
      q: "၁၆။ Canny မှ ထွက်လာသော မျဉ်းကြောင်းများကို အဆက်အစပ်ရှိသော ပုံသဏ္ဌာန် (Object) တစ်ခုအဖြစ် ပေါင်းစည်းခြေရာခံပေးသော လုပ်ငန်းစဉ်ကို မည်သို့ခေါ်သနည်း?",
      options: ["Color Masking", "Thresholding", "Find Contours", "Feature Extraction"],
      ans: 2
    },
    {
      q: "၁၇။ Deep Learning ခေတ် မတိုင်မီက မျက်နှာရှာဖွေခြင်း (Face Detection) တွင် အဓိက အသုံးပြုခဲ့သော အယ်လ်ဂိုရီသမ်မှာ အဘယ်နည်း?",
      options: ["YOLO", "Haar Cascades", "ResNet", "U-Net"],
      ans: 1
    },
    {
      q: "၁၈။ အလွှာများစွာပါဝင်သော CNN Model တစ်ခု၏ အစောပိုင်းအလွှာ (Early Layers) များသည် ပုံထဲမှ မည်သည့်အရာများကို အဓိက သင်ယူလေ့ရှိသနည်း?",
      options:["လူမျက်နှာ၊ ကား စသည့် အရာဝတ္ထုအပြည့်အစုံ", "အနားသတ်မျဉ်းများ၊ လိုင်းများ (Low-level features)", "ပုံ၏ နောက်ဆုံး Classification ရာခိုင်နှုန်း", "စာသားများ"],
      ans: 1
    },
    {
      q: "၁၉။ Neural Network တစ်ခု၏ အခြေခံအကျဆုံး တွက်ချက်မှု ယူနစ် (Artificial Neuron) ကို မည်သို့ ခေါ်သနည်း?",
      options: ["Perceptron", "Transformer", "Kernel", "Bounding Box"],
      ans: 0
    },
    {
      q: "၂၀။ Linear Transformation တွက်ချက်ရာတွင် အသုံးပြုသော ဖော်မြူလာမှာ အဘယ်နည်း?",
      options: ["(X * W) + b", "(X + W) * b", "X^2 + W", "(W * b) + X"],
      ans: 0
    },
    {
      q: "၂၁။ ယနေ့ခေတ် CNN များ၏ Hidden Layers များတွင် Vanishing Gradient ပြဿနာကို ကာကွယ်ရန် အများဆုံး အသုံးပြုသော Activation Function မှာ အဘယ်နည်း?",
      options: ["Sigmoid", "Softmax", "Tanh", "ReLU (Rectified Linear Unit)"],
      ans: 3
    },
    {
      q: "၂၂။ ReLU Function ၏ အလုပ်လုပ်ပုံမှာ အဘယ်နည်း?",
      options:["အနှုတ်ဂဏန်းများကို 0 အဖြစ်ပြောင်း၍၊ အပေါင်းဂဏန်းများကို မူလအတိုင်းထားသည်", "ဂဏန်းအားလုံးကို 0 နှင့် 1 ကြား ပြောင်းပေးသည်", "ဂဏန်းအားလုံးကို ပေါင်းလျှင် 1 ဖြစ်အောင် ညှိပေးသည်", "အပေါင်းဂဏန်းများကို 0 ပြောင်းပေးသည်"],
      ans: 0
    },
    {
      q: "၂၃။ Multi-class Classification ပြဿနာများတွင် နောက်ဆုံး Output Layer ၌ ရလဒ်များပေါင်းလျှင် 1 (100%) ကွက်တိဖြစ်အောင် ပြုလုပ်ပေးသော Function မှာ အဘယ်နည်း?",
      options: ["ReLU", "Softmax", "Sigmoid", "Cross-Entropy"],
      ans: 1
    },
    {
      q: "၂၄။ AI Model ၏ အဖြေနှင့် အဖြေမှန်ကြားရှိ လွဲချော်မှု ပမာဏကို တိုင်းတာပေးသော ပုံသေနည်းကို မည်သို့ခေါ်သနည်း?",
      options: ["Activation Function", "Optimization", "Loss Function", "Gradient Descent"],
      ans: 2
    },
    {
      q: "၂၅။ Model အဖြေမှားသွားသောအခါ အမှားသည် မည်သည့် Weight ကြောင့် ဖြစ်သည်ဆိုသည်ကို နောက်ပြန်လှည့်၍ တွက်ချက်သော လုပ်ငန်းစဉ်ကို မည်သို့ခေါ်သနည်း?",
      options: ["Forward Pass", "Backpropagation", "Transfer Learning", "Data Augmentation"],
      ans: 1
    },
    {
      q: "၂၆။ Computer Vision နယ်ပယ်တွင် ယနေ့ခေတ် အများဆုံးအသုံးပြုသော Optimizer မှာ အဘယ်နည်း?",
      options: ["SGD", "Adam", "RMSProp", "Adagrad"],
      ans: 1
    },
    {
      q: "၂၇။ PyTorch တွင် AI ကို Train သောအခါ Tensors များ၏ Dimension ပုံစံ (Shape) အစီအစဉ်မှာ အဘယ်နည်း?",
      options: ["(Batch, Channels, Height, Width)", "(Height, Width, Channels)", "(Batch, Height, Width, Channels)", "(Channels, Batch, Height, Width)"],
      ans: 0
    },
    {
      q: "၂၈။ ပုံရိပ်များကို 1D Array သို့ မဆွဲဆန့်ဘဲ 2D အတိုင်း တွက်ချက်နိုင်သဖြင့် Spatial Information ကို မပျက်စီးစေသော Neural Network အမျိုးအစားမှာ အဘယ်နည်း?",
      options: ["RNN", "LSTM", "CNN", "Simple ANN"],
      ans: 2
    },
    {
      q: "၂၉။ CNN တွင် Max Pooling အလွှာကို ထည့်သွင်းအသုံးပြုခြင်း၏ အဓိက ရည်ရွယ်ချက်မှာ အဘယ်နည်း?",
      options:["ပုံကို ပိုမိုကြည်လင်ပြတ်သားလာစေရန်", "အရေးကြီးသော Feature များကို ချန်ထားခဲ့ပြီး ပုံ၏ အရွယ်အစားကို လျှော့ချရန်", "အရာဝတ္ထုများ၏ အရောင်ကို ပြောင်းလဲပေးရန်", "3D Tensor မှ 1D Array သို့ ပြောင်းလဲပေးရန်"],
      ans: 1
    },
    {
      q: "၃၀။ CNN ၏ 3D Feature Map များကို နောက်ဆုံး ဆုံးဖြတ်ချက်ချမည့် Fully Connected Layer များ နားလည်စေရန် 1D အဖြစ် ဆွဲဆန့်ပေးသော လုပ်ငန်းစဉ်ကို မည်သို့ခေါ်သနည်း?",
      options: ["Padding", "Flattening", "Pooling", "Striding"],
      ans: 1
    },
    {
      q: "၃၁။ ResNet ၏ အကြီးမားဆုံး တီထွင်မှုဖြစ်ပြီး၊ အလွှာပေါင်း ရာနှင့်ချီ၍ Train သည့်တိုင်အောင် Gradient များ မပျောက်ဆုံးစေရန် ကူညီပေးသော နည်းပညာမှာ အဘယ်နည်း?",
      options: ["Skip Connections", "Flattening", "Data Augmentation", "Softmax"],
      ans: 0
    },
    {
      q: "၃၂။ ImageNet ကဲ့သို့ Dataset ကြီးများဖြင့် Train ထားသော Model များကို ယူ၍ မိမိ၏ သီးသန့် Project များအတွက် ပြန်လည်အသုံးပြုခြင်းကို မည်သို့ခေါ်သနည်း?",
      options: ["Reinforcement Learning", "Unsupervised Learning", "Transfer Learning", "Imitation Learning"],
      ans: 2
    },
    {
      q: "၃၃။ Transfer Learning အသုံးပြုရာတွင် မူလ Pre-trained Model ၏ အစောပိုင်း Layers များကို Weight မပြောင်းလဲစေရန် မည်သို့ ပြုလုပ်ရသနည်း?",
      options:["Flatten လုပ်ပစ်လိုက်သည်", "Layer များကို ဖျက်ပစ်လိုက်သည်", "Gradient တွက်ချက်မှုကို ပိတ်ထားသည် (Freeze လုပ်သည်)", "Learning Rate ကို အလွန်မြှင့်ထားလိုက်သည်"],
      ans: 2
    },
    {
      q: "၃၄။ AI သည် ပုံများကို အလွတ်ကျက်သွားခြင်း (Overfitting) မဖြစ်စေရန် ပုံများကို လှည့်ခြင်း၊ ပြောင်းပြန်လှန်ခြင်း စသည်တို့ ပြုလုပ်ပေးသော နည်းလမ်းကို မည်သို့ခေါ်သနည်း?",
      options: ["Data Normalization", "Data Augmentation", "Data Collection", "Data Annotation"],
      ans: 1
    },
    {
      q: "၃၅။ Object Detection သည် Image Classification နှင့် မည်သည့်အချက် ကွာခြားသနည်း?",
      options:["Classification ထက် ပိုမြန်သည်", "အရာဝတ္ထု၏ အမျိုးအစား (Class) သာမက၊ တည်နေရာ (Bounding Box) ကိုပါ ရှာဖွေပေးသည်", "ပုံတစ်ပုံတွင် အရာဝတ္ထုတစ်ခုတည်းကိုသာ ရှာဖွေပေးနိုင်သည်", "Accuracy ပိုများသည်"],
      ans: 1
    },
    {
      q: "၃၆။ YOLO (You Only Look Once) ကဲ့သို့ Detector များကို မည်သည့် အမျိုးအစားဟု ခေါ်သနည်း?",
      options: ["Two-Stage Detectors", "One-Stage Detectors", "End-to-End Detectors", "Zero-Shot Detectors"],
      ans: 1
    },
    {
      q: "၃၇။ Bounding Box နှစ်ခု မည်မျှ တိကျစွာ ထပ်တူကျသည်ကို တိုင်းတာသော ပုံသေနည်း (Metric) ကို မည်သို့ခေါ်သနည်း?",
      options: ["mAP", "IoU (Intersection over Union)", "NMS", "F1-Score"],
      ans: 1
    },
    {
      q: "၃၈။ One-Stage Detector များတွင် အရာဝတ္ထု တစ်ခုတည်းအပေါ်၌ Bounding Box များ ထပ်နေခြင်း (Duplicate Boxes) ကို ဖယ်ရှားရှင်းလင်းပေးသော လုပ်ငန်းစဉ်ကို မည်သို့ခေါ်သနည်း?",
      options: ["IoU", "mAP", "NMS (Non-Maximum Suppression)", "SGD"],
      ans: 2
    },
    {
      q: "၃၉။ Object Detection မော်ဒယ်များ၏ စွမ်းဆောင်ရည်ကို တိုင်းတာသော အမြင့်ဆုံး စံချိန်စံညွှန်း Metric မှာ အဘယ်နည်း?",
      options: ["Accuracy", "MSE Loss", "mAP (Mean Average Precision)", "Cross-Entropy"],
      ans: 2
    },
    {
      q: "၄၀။ Image Classification တွင် မော်ဒယ်၏ ခန့်မှန်းချက်များအနက် မှန်ကန်သော ခန့်မှန်းချက်ရာခိုင်နှုန်း စုစုပေါင်းကို တိုင်းတာသော Metric ကို မည်သို့ခေါ်သနည်း?",
      options: ["Precision", "Recall", "Accuracy", "F1-Score"],
      ans: 2
    },
    {
      q: "၄၁။ YOLO ကို Custom Train သည့်အခါ ပုံရှိ အရာဝတ္ထုများကို Bounding Box ရေးဆွဲပေးရသော လုပ်ငန်းစဉ်ကို မည်သို့ခေါ်သနည်း?",
      options: ["Data Masking", "Data Augmentation", "Data Annotation (Labeling)", "Data Scraping"],
      ans: 2
    },
    {
      q: "၄၂။ Custom Object Detection Model ကို Train ရာတွင် Dataset ၏ လမ်းကြောင်းနှင့် Class အမည်များကို ကြေညာပေးရသော အရေးအကြီးဆုံး ဖိုင်မှာ အဘယ်နည်း?",
      options: ["requirements.txt", "data.yaml", "train.py", "weights.pt"],
      ans: 1
    },
    {
      q: "၄၃။ Bounding Box အစား ပုံထဲရှိ Pixel တစ်ခုချင်းစီကို တိကျစွာ ခွဲခြားအရောင်ခြယ်ပေးသော နည်းပညာကို မည်သို့ခေါ်သနည်း?",
      options: ["Image Classification", "Image Segmentation", "Image Generation", "Object Tracking"],
      ans: 1
    },
    {
      q: "၄၄။ AI က 'ဟုတ်သည် (Positive)' ဟု ခန့်မှန်းလိုက်သော အဖြေများထဲတွင် တကယ်အမှန်တကယ် 'ဟုတ်သည်' ဟု ကိုက်ညီမှုရာခိုင်နှုန်းကို တိုင်းတာသော အသုံးအနှုန်း (Term) မှာ အဘယ်နည်း?",
      options: ["Recall", "Precision", "Accuracy", "Threshold"],
      ans: 1
    },
    {
      q: "၄၅။ တကယ့်အဖြေမှန် (Actual Positives) အားလုံးထဲမှ AI က မည်မျှကို အမှန်တကယ် ရှာဖွေတွေ့ရှိခဲ့သနည်း (လွတ်မသွားဘဲ ရှာတွေ့မှုရာခိုင်နှုန်း) ကို တိုင်းတာသော Metric မှာ အဘယ်နည်း?",
      options: ["Precision", "Recall", "Accuracy", "mAP"],
      ans: 1
    },
    {
      q: "၄၆။ Precision နှင့် Recall နှစ်ခုလုံးကို မျှတစွာ တွက်ချက်လိုသောအခါ ၎င်းတို့နှစ်ခု၏ Harmonic Mean အဖြစ် အသုံးပြုသော Evaluation Metric မှာ အဘယ်နည်း?",
      options: ["IoU", "Accuracy", "F1-Score", "Confusion Matrix"],
      ans: 2
    },
    {
      q: "၄၇။ Classification Model တစ်ခု၏ မှန်ကန်မှုနှင့် မှားယွင်းမှု (True Positive, False Positive စသည်) တို့ကို အသေးစိတ် ဇယားကွက်ဖြင့် ပြသပေးသော အရာကို မည်သို့ခေါ်သနည်း?",
      options: ["Correlation Matrix", "Confusion Matrix", "Bounding Box", "Feature Map"],
      ans: 1
    },
    {
      q: "၄၈။ ပုံထဲတွင် ခွေး တကယ်မရှိဘဲလျက် AI က 'ခွေးရှိသည်' ဟု မှားယွင်းစွာ ခန့်မှန်းလိုက်ခြင်းကို မည်သည့် Term ဖြင့် ခေါ်ဆိုသနည်း?",
      options: ["True Positive", "True Negative", "False Positive", "False Negative"],
      ans: 2
    },
    {
      q: "၄၉။ ပုံထဲတွင် ခွေး တကယ်ရှိနေသော်လည်း AI က မရှာတွေ့ဘဲ လွတ်သွားခြင်း (Miss ဖြစ်ခြင်း) ကို မည်သည့် Term ဖြင့် ခေါ်ဆိုသနည်း?",
      options: ["True Positive", "True Negative", "False Positive", "False Negative"],
      ans: 3
    },
    {
      q: "၅၀။ Object Detection တွင် AI မှ ရှာဖွေတွေ့ရှိသော Bounding Box တစ်ခု၏ တည်နေရာကို ကိုယ်စားပြုရန် မည်သည့် ဂဏန်းတန်ဖိုးများကို အသုံးပြုလေ့ရှိသနည်း?",
      options:["ပုံ၏ Pixel အရေအတွက်", "ပုံ၏ Width နှင့် Height သီးသန့်", "(x_min, y_min, x_max, y_max) သို့မဟုတ် (x_center, y_center, width, height)", "(x, y, z) 3D Coordinates များ"],
      ans: 2
    }
  ];

  // မေးခွန်းများကို Form ထဲသို့ ထည့်သွင်းခြင်း
  questionsData.forEach(function(q, index) {
    var item = form.addMultipleChoiceItem();
    item.setTitle(q.q);
    item.setPoints(1); // မေးခွန်းတစ်ခုလျှင် ၁ မှတ်
    
    // ရွေးချယ်စရာ Options များကို ဖန်တီးပြီး အဖြေမှန်ကို သတ်မှတ်ခြင်း
    var choices = q.options.map(function(opt, i) {
      return item.createChoice(opt, i === q.ans);
    });
    
    item.setChoices(choices);
  });

  // စာမေးပွဲ ဖန်တီးပြီးကြောင်း လင့်ခ်ကို Log တွင် ပြသခြင်း
  Logger.log("Quiz Successfully Created!");
  Logger.log("Edit URL: " + form.getEditUrl());
  Logger.log("Published URL: " + form.getPublishedUrl());
}