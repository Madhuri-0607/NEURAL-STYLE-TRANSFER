COMPANY: CODTECH IT SOLUTIONS   
NAME: MOTHUKURI MADHURI    
INTERN ID: C0DF104   
DOMAIN: Artificial Intelligence Markup Language (AIML Internship)   
DURATION: 4 WEEKS   
MENTOR: NEELA SANTHOSH   


# NEURAL-STYLE-TRANSFER

This project implements a Neural Style Transfer algorithm using PyTorch. It blends the content of one image with the artistic style of another by optimizing a target image to match content and style features extracted from a pretrained VGG19 model.

📌 Features 
* Transfers artistic style from one image onto another 

* Uses a pretrained VGG19 model for feature extraction 

* Supports custom content and style images 

* Optimized with Adam optimizer and custom loss functions 

* Outputs a stylized image after training 

🖼️ Example

Input:

Content Image: ![content](https://github.com/user-attachments/assets/5c4d10ad-80e7-4ab8-88db-8579f9e8bb66)

Style Image: ![style (2)](https://github.com/user-attachments/assets/833fc2fa-d3af-4378-96ee-be9ef212001a)

Output:

Stylized Image: ![WhatsApp Image 2025-05-19 at 15 07 18_f0cec0ad](https://github.com/user-attachments/assets/6bb29aa5-f481-4c39-b31a-f17670133bf4)

🚀 Getting Started 
🔧 Prerequisites 
* Python 3.8+ 

* PyTorch 

* torchvision 

* matplotlib 

* Pillow 

📥 Installation
Clone the repository: 

git clone https://github.com/Madhuri-0607/neural-style-transfer.git   
cd neural-style-transfer 

Install dependencies: 

pip install -r requirements.txt  

▶️ Run the Script 

python style_transfer.py 

⚙️ How It Works 
* The content and style images are loaded and preprocessed. 

* A pretrained VGG19 model is used to extract features from specific layers. 

* The content loss and style loss are computed. 

* The target image (initially a copy of the content image) is optimized to minimize the total loss. 

* The result is displayed and saved. 

📁 Directory Structure  

.  
├── style_transfer.py   
├── requirements.txt   
├── content.jpg   
├── style.jpg   
└── output.jpg   

📚 References 
Gatys et al., "Image Style Transfer Using Convolutional Neural Networks", CVPR 2016 

PyTorch documentation: https://pytorch.org/docs/stable/index.html  

📄 License 
This project is licensed under the MIT License. 




