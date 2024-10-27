import streamlit as st
import torch 
import os
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
weights = ResNet50_Weights.DEFAULT
resnet50_transformers = weights.transforms()
model= resnet50()

device='cuda' if torch.cuda.is_available() else 'cpu'

saved_model=torch.load(r'mask_detection_resnet60_50epochs_model.pth',
                       map_location=torch.device(device))

model.load_state_dict(saved_model)

from typing import Tuple, Dict
from timeit import default_timer as timer
class_names=['mask', 'mask incorrect', 'without mask']
def predict(img,
            model:torch.nn.Module,
            num_class,
            transforms)-> Tuple[Dict,float]:
  # Starting a timer
  start_time=timer()
  # Transform the input image for use with EffnetB2
  transformed_img=transforms(img).unsqueeze(0)
  # Putting model into eval mode, make prediction
  model.eval()
  with torch.inference_mode():
    logits=model(transformed_img)
    probs=torch.softmax(logits,dim=1)
    final_pred=class_names[torch.argmax(probs,dim=1)]

  # Creating a prediction label and prediction probablity dictionary
  pred_labels_and_probs={class_names[i]: float(probs[0][i]) for i in range(len(num_class))}

  # Calculating pred time
  end_time=timer()
  pred_time=round(end_time-start_time,4)

  return pred_labels_and_probs,pred_time,final_pred



st.set_page_config(page_title="Mask Detection")
st.title("Mask Detection with Pytorch")
st.write("Please Select your JPG file or Enable Camera")

image_source=st.radio("Select Image Source:",("Take Picture","Upload Image"))
if image_source == "Take Picture":
    uploaded_image = st.camera_input("Take a picture:")
else:
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])


if uploaded_image:
    img=Image.open(uploaded_image)
    pred_dict,pred_time,final_pred=predict(img,
                                model=model,
                                num_class=class_names,
                                transforms=resnet50_transformers)
    st.write(f"Resutl : ",final_pred)
    st.write(f"Probality : ",pred_dict)
    st.write("Time Taken : " ,pred_time )
else:
    st.write("Please Give your Input Picture")    
