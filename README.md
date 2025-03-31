# 3D Reference Image Generator for Artists 🎨📸

Artists often spend hours looking for just the right pose, lighting, or angle—especially for rare concepts like **"a woman wearing a suit playing baseball."** Traditional image search tools rely on real-world photos, which can be limiting.

This project offers a smarter solution: **a prototype tool that generates personalized reference images using AI-powered 3D visual search.**

---

## 🔍 What It Does

- Transforms a simple input image into a photorealistic, fully stylized version guided by a **depth map**
- Uses **Stable Diffusion + ControlNet** to control structure and detail
- Helps generate poses and compositions hard to find in real photos

---

## 🛠 How It Works

1. **Depth Estimation**  
   A pretrained pipeline extracts depth information from the input image

2. **ControlNet Integration**  
   Feeds the depth map into a ControlNet-powered Stable Diffusion Img2Img pipeline

3. **Image Generation**  
   Outputs a high-quality, stylized image based on your custom text prompt
