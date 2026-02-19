import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms, datasets
from efficientnet_pytorch import EfficientNet

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================== CẤU HÌNH FLASK ==================
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
GRADCAM_FOLDER = os.path.join("static", "gradcam")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER

# ================== DEVICE ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================== LẤY DANH SÁCH LỚP ==================
TRAIN_DIR = r"E:\archive\SkinDisease\SkinDisease\train"   # sửa nếu khác

IMAGE_SIZE = 300
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

tmp_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=tmp_tf)
class_names = train_ds.classes
num_classes = len(class_names)

print("Số lớp:", num_classes)
print("Classes:", class_names)

# ================== TRANSFORMS ==================
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

display_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ================== HÀM LOAD MODEL ==================
def load_vgg16(num_classes, ckpt_path):
    try:
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
    except AttributeError:
        model = models.vgg16(pretrained=True)

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    print("✅ VGG16 loaded.")
    return model

def load_resnet50(num_classes, ckpt_path):
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    except AttributeError:
        model = models.resnet50(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    print("✅ ResNet50 loaded.")
    return model

def load_efficientnet_b4(num_classes, ckpt_path):
    model = EfficientNet.from_pretrained("efficientnet-b4", num_classes=num_classes)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    print("✅ EfficientNet-B4 loaded.")
    return model

def load_efficientnet_b0(num_classes, ckpt_path):
    model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=num_classes)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    print("✅ EfficientNet-B0 loaded.")
    return model

def load_densenet121(num_classes, ckpt_path):
    try:
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights)
    except AttributeError:
        model = models.densenet121(pretrained=True)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    print("✅ DenseNet121 loaded.")
    return model

def load_mobilenet_v3(num_classes, ckpt_path):
    try:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_large(weights=weights)
    except AttributeError:
        model = models.mobilenet_v3_large(pretrained=True)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    print("✅ MobileNetV3-Large loaded.")
    return model

# ========= ĐƯỜNG DẪN CHECKPOINT (SỬA CHO ĐÚNG FILE CỦA BẠN) =========
VGG_CKPT_PATH      = r"E:\archive\PiplineCode pretrain=True\vgg16_pretrained_fulloption.pth"
RES50_CKPT_PATH    = r"E:\archive\PiplineCode pretrain=True\resnet50_pretrained_fulloption.pth"
EFFB4_CKPT_PATH    = r"E:\archive\PiplineCode pretrain=True\efficientnet_b4_pretrained_fulloption.pth"
EFFB0_CKPT_PATH    = r"E:\archive\PiplineCode pretrain=True\efficientnet_b0_pretrained_fulloption.pth"
DENSE121_CKPT_PATH = r"E:\archive\PiplineCode pretrain=True\densenet121_pretrained_fulloption.pth"
MOBILEV3_CKPT_PATH = r"E:\archive\PiplineCode pretrain=True\mobilenetv3_fulloption_imagenet.pth"

# ========= KHỞI TẠO 6 MODEL =========
vgg16_model     = load_vgg16(num_classes, VGG_CKPT_PATH)
resnet50_model  = load_resnet50(num_classes, RES50_CKPT_PATH)
effb4_model     = load_efficientnet_b4(num_classes, EFFB4_CKPT_PATH)
effb0_model     = load_efficientnet_b0(num_classes, EFFB0_CKPT_PATH)
densenet_model  = load_densenet121(num_classes, DENSE121_CKPT_PATH)
mobilenet_model = load_mobilenet_v3(num_classes, MOBILEV3_CKPT_PATH)

print("✅ Loaded all 6 models.")

# ================== GRAD-CAM IMPLEMENTATION ==================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fh = target_layer.register_forward_hook(forward_hook)
        self.bh = target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        else:
            target_class = int(target_class)

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0

        output.backward(gradient=one_hot)

        grads = self.gradients        # [1, C, H, W]
        acts  = self.activations      # [1, C, H, W]

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * acts).sum(dim=1)               # [1, H, W]
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cam.squeeze().cpu().numpy()               # [H, W]
        return cam

    def remove_hooks(self):
        self.fh.remove()
        self.bh.remove()

def get_target_layer(model, model_name):
    if model_name == "vgg16":
        return model.features[-1]
    elif model_name == "resnet50":
        return model.layer4[-1]
    elif model_name in ["efficientnet_b4", "efficientnet_b0"]:
        return model._conv_head
    elif model_name == "densenet121":
        return model.features[-1]
    elif model_name == "mobilenetv3":
        return model.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")

def generate_gradcam_for_model(model, model_name, img_tensor, original_img, class_idx):
    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(img_tensor, target_class=class_idx)
    gradcam.remove_hooks()

    cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(original_img_bgr, 0.5, heatmap, 0.5, 0)

    return superimposed

def save_gradcam_image(gradcam_img, filename):
    save_path = os.path.join(app.config["GRADCAM_FOLDER"], filename)
    cv2.imwrite(save_path, gradcam_img)
    return url_for("static", filename=f"gradcam/{filename}")

# ================== HÀM DỰ ĐOÁN ==================
def predict_single_model(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        probs_np = probs.cpu().numpy()[0]
        idx = int(np.argmax(probs_np))
        prob = float(probs_np[idx])
    return idx, prob, probs_np

def soft_voting_predict_pil(
    pil_img,
    vgg_model,
    res_model,
    effb4_model,
    effb0_model,
    dense_model,
    mobile_model,
    weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
):
    start_time = time.time()

    x = inference_transform(pil_img).unsqueeze(0).to(device)
    original_img_np = np.array(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)))

    # ---- Dự đoán từng model ----
    vgg_idx, vgg_prob, vgg_vec   = predict_single_model(vgg_model, x)
    res_idx, res_prob, res_vec   = predict_single_model(res_model, x)
    eff4_idx, eff4_prob, eff4_vec= predict_single_model(effb4_model, x)
    eff0_idx, eff0_prob, eff0_vec= predict_single_model(effb0_model, x)
    den_idx, den_prob, den_vec   = predict_single_model(dense_model, x)
    mob_idx, mob_prob, mob_vec   = predict_single_model(mobile_model, x)

    # ---- Soft voting 6 model ----
    w_vgg, w_res, w_eff4, w_eff0, w_den, w_mob = weights
    total_w = w_vgg + w_res + w_eff4 + w_eff0 + w_den + w_mob

    avg_probs = (
        w_vgg * vgg_vec +
        w_res * res_vec +
        w_eff4 * eff4_vec +
        w_eff0 * eff0_vec +
        w_den * den_vec +
        w_mob * mob_vec
    ) / total_w

    final_idx = int(np.argmax(avg_probs))
    final_prob = float(avg_probs[final_idx])

    # ---- Grad-CAM cho từng model ----
    gradcam_urls = {}
    try:
        timestamp = str(int(time.time() * 1000))

        vgg_cam = generate_gradcam_for_model(vgg_model, "vgg16", x, original_img_np, vgg_idx)
        gradcam_urls["vgg"] = save_gradcam_image(vgg_cam, f"gradcam_vgg_{timestamp}.jpg")

        res_cam = generate_gradcam_for_model(res_model, "resnet50", x, original_img_np, res_idx)
        gradcam_urls["resnet50"] = save_gradcam_image(res_cam, f"gradcam_resnet_{timestamp}.jpg")

        eff4_cam = generate_gradcam_for_model(effb4_model, "efficientnet_b4", x, original_img_np, eff4_idx)
        gradcam_urls["effb4"] = save_gradcam_image(eff4_cam, f"gradcam_effb4_{timestamp}.jpg")

        eff0_cam = generate_gradcam_for_model(effb0_model, "efficientnet_b0", x, original_img_np, eff0_idx)
        gradcam_urls["effb0"] = save_gradcam_image(eff0_cam, f"gradcam_effb0_{timestamp}.jpg")

        den_cam = generate_gradcam_for_model(dense_model, "densenet121", x, original_img_np, den_idx)
        gradcam_urls["densenet"] = save_gradcam_image(den_cam, f"gradcam_dense_{timestamp}.jpg")

        mob_cam = generate_gradcam_for_model(mobile_model, "mobilenetv3", x, original_img_np, mob_idx)
        gradcam_urls["mobilenet"] = save_gradcam_image(mob_cam, f"gradcam_mobile_{timestamp}.jpg")

    except Exception as e:
        print(f"❌ Lỗi khi tạo Grad-CAM: {e}")
        gradcam_urls = {
            "vgg":       url_for("static", filename="placeholder.jpg"),
            "resnet50":  url_for("static", filename="placeholder.jpg"),
            "effb4":     url_for("static", filename="placeholder.jpg"),
            "effb0":     url_for("static", filename="placeholder.jpg"),
            "densenet":  url_for("static", filename="placeholder.jpg"),
            "mobilenet": url_for("static", filename="placeholder.jpg"),
        }

    end_time = time.time()
    prediction_time = round(end_time - start_time, 2)

    return {
        "vgg": {
            "label": class_names[vgg_idx],
            "prob": vgg_prob
        },
        "resnet50": {
            "label": class_names[res_idx],
            "prob": res_prob
        },
        "effb4": {
            "label": class_names[eff4_idx],
            "prob": eff4_prob
        },
        "effb0": {
            "label": class_names[eff0_idx],
            "prob": eff0_prob
        },
        "densenet": {
            "label": class_names[den_idx],
            "prob": den_prob
        },
        "mobilenet": {
            "label": class_names[mob_idx],
            "prob": mob_prob
        },
        "soft_vote": {
            "label": class_names[final_idx],
            "prob": final_prob
        },
        "gradcam_urls": gradcam_urls,
        "prediction_time": prediction_time
    }

# ================== ROUTES WEB ==================
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    image_url = None
    error = None
    gradcam_urls = None
    prediction_time = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "Không tìm thấy file ảnh."
        else:
            file = request.files["image"]
            if file.filename == "":
                error = "Bạn chưa chọn ảnh."
            else:
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(save_path)

                pil_img = Image.open(save_path).convert("RGB")
                prediction_result = soft_voting_predict_pil(
                    pil_img,
                    vgg_model=vgg16_model,
                    res_model=resnet50_model,
                    effb4_model=effb4_model,
                    effb0_model=effb0_model,
                    dense_model=densenet_model,
                    mobile_model=mobilenet_model,
                    weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
                )

                results = {
                    "vgg":       prediction_result["vgg"],
                    "resnet50":  prediction_result["resnet50"],
                    "effb4":     prediction_result["effb4"],
                    "effb0":     prediction_result["effb0"],
                    "densenet":  prediction_result["densenet"],
                    "mobilenet": prediction_result["mobilenet"],
                    "soft_vote": prediction_result["soft_vote"],
                }
                image_url = url_for("static", filename=f"uploads/{filename}")
                gradcam_urls = prediction_result.get("gradcam_urls")
                prediction_time = prediction_result.get("prediction_time")

                print("GradCAM URLs:", gradcam_urls)

    return render_template(
        "index.html",
        results=results,
        image_url=image_url,
        error=error,
        gradcam_urls=gradcam_urls,
        prediction_time=prediction_time
    )

# ================== TẠO PLACEHOLDER IMAGE ==================
def create_placeholder_image():
    """Tạo ảnh placeholder nếu cần (phòng khi Grad-CAM lỗi)."""
    placeholder_path = os.path.join("static", "placeholder.jpg")
    if not os.path.exists(placeholder_path):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img.fill(240)
        cv2.putText(img, "Grad-CAM", (70, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        cv2.putText(img, "Placeholder", (70, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        cv2.imwrite(placeholder_path, img)

# ================== CHẠY APP ==================
if __name__ == "__main__":
    create_placeholder_image()
    print("✅ Grad-CAM ready!")
    print("✅ Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
