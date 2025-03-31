3D Reference Image Generator for Artists 🎨📸

Artists often spend hours looking for just the right pose, lighting, or angle—especially for rare concepts like “a woman wearing a suit playing baseball.” Traditional image search tools rely on real-world photos, which can be limiting.

This project provides a smarter solution: a prototype tool that generates personalized reference images using AI-powered 3D visual search.

🔍 What It Does
	•	Transforms a simple input image into a photorealistic, fully stylized version guided by a depth map.
	•	Uses Stable Diffusion + ControlNet to control structure and detail.
	•	Supports generating poses and compositions that are hard to find in real photos.

🛠 How It Works
	1.	Depth Estimation: Uses a pretrained depth estimation pipeline to extract structure from an image.
	2.	ControlNet Integration: Feeds this depth map into the ControlNet-enabled Stable Diffusion Img2Img pipeline.
	3.	Image Generation: Outputs a high-quality, stylized image based on your prompt.

📁 Files & Usage
	•	Load your input image from Google Drive.
	•	Generate the depth map using depth-estimation.
	•	Run the pipeline with a custom prompt to generate your new reference image.

💡 Example Prompt

"a photorealistic female model in an action pose, dramatic studio lighting, wearing a dress, standing in a clean studio environment, high detail, full body"

🔧 Dependencies
	•	Python
	•	PyTorch
	•	Hugging Face Transformers & Diffusers
	•	PIL
	•	Google Colab (for running with GPU)

🚀 Future Directions
	•	Add support for edge maps (Canny, HED).
	•	Build an interactive UI for pose + outfit selection.
	•	Train on custom datasets to improve generalization.
