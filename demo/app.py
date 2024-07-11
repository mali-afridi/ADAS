# import streamlit as st
# from PIL import Image
# import cv2
# import numpy as np
# import os
# import imghdr
# import pandas as pd
# import pdb 
# from sami import inf

# # from Unet_I import binary_unet
# # from Unet_II import inpaint_unet

# st.title("Transformer Based ADAS")
# st.write("---")

# sidebar_options = ["Project Info","Inference","Demo Image","Upload your Image","Training Analysis"]
# st.sidebar.success("Advanced Driving Assitance System")
# st.sidebar.write('---')
# st.sidebar.title("Options")
# box = st.sidebar.selectbox(" ", sidebar_options)

# # Inference
# def run_demo(name):
#     # st.image("demo_imgs/gt_imgs/"+name+".jpg",width=450,caption="Ground Truth")
#     st.image("80.jpg",width=450,caption="80.JPG")
#     col1,col2 = st.columns(2)
#     with col1:
#         st.image("demo.jpg",caption="demo.JPG")
#     with col2:
#         st.image("20.jpg",caption="20.JPG")
#     st.image("ali.png",width=450,caption="Detection")

# if box == "Project Info":

#     name = "80.jpg"
#     col1,col2= st.columns(2)
#     with col1:
#         st.image(name)
#         st.text(f"{name}")
        
#     with col2:
#         st.image("20.jpg")        
#         st.markdown(" $~$\n\n$~$\n\n $~~~~~~~~~~~~~~$ 20.JPG")

    
#     st.markdown("This project attempted to achieve the paper **[A novel GAN-based network for unmasking of "
#                 "masked face](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9019697)**. The model "
#                 "is designed to remove the face-mask from facial image and inpaint the left-behind region based "
#                 "on a novel GAN-network approach. ")
#     # st.image("info_imgs/md_archi.png")
#     st.markdown("Rather than using the traditional pix2pix U-Net method, in this work the model consists of two main modules, "
#                 "map module and editing module. In the first module, we detect the face-mask object and generate a "
#                 "binary segmentation map for data augmentation. In the second module, we train the modified U-Net "
#                 "with two discriminators using masked image and binary segmentation map.")
#     st.markdown("***Feel free to play it around:***")
#     st.markdown(":point_left: To get started, you can choose ***Demo Image*** to see the performance of the model.")
#     st.markdown(":camera: Feel free to ***upload*** any masked image you want and see the performance.")
#     st.markdown(":chart_with_upwards_trend: Also, press ***Training Analysis*** to see the training insight.")


# elif box=="Inference":
#     fps, objects, lanes = 0, 0, 0
#     st.sidebar.info('Click on the ***button*** to run the Inference on the model using the ***pre-defined dataloader***.')
    
#     st.subheader("Inference on Dataloader")
#     st.write("This uses the pre-defined Dataloader for the inference. The dataloader consists of the Validation images of the CULane Dataset.")
#     st.button("Inference!")
    
#     st.write("---")

#     st.write("---")
    
#     st.subheader("Live Inference Stats!")

#     c1, c2, c3 = st.columns(3)

#     with c1:
#         st.write("FPS: ", fps)
#         st.write("Objects: ", objects)
#         st.write("Lanes: ", lanes)

#     with c2:
#         st.write("FPS: ", fps)
#         st.write("Objects: ", objects)
#         st.write("Lanes: ", lanes)

#     with c3:
#         st.write("FPS: ", fps)
#         st.write("Objects: ", objects)
#         st.write("Lanes: ", lanes)

# elif box == "Demo Image":
#     st.sidebar.write("---")

#     demoimg_dir = os.getcwd()
#     photos=[]
#     for file in os.listdir(demoimg_dir):
#         filepath = os.path.join(demoimg_dir,file)
#         if imghdr.what(filepath) is not None:
#             photos.append(file[:-4])
#     photos.sort()

#     inpaint_option = st.sidebar.selectbox("Please select a sample image, then click the 'Inpaint!' button.",photos)
#     inpaint = st.sidebar.button("Inpaint !")

#     if inpaint:
#         st.empty()
#         run_demo(inpaint_option)


# elif box == "Upload your Image":
#     st.sidebar.info('Please upload ***single masked person*** image. For best result, please also ***center the face*** in the image, and the face mask should be in ***light green/blue color***.')
#     image = st.file_uploader("Upload your masked image here",type=['jpg','png','jpeg'])
#     if image is not None:
#         col1,col2 = st.columns(2)
#         masked = Image.open(image).convert('RGB')
#         masked = np.array(masked)
#         masked = cv2.resize(masked,(224,224))
#         with col1:
#             st.image(masked,width=300,caption="masked photo")
#         with col2:
#             st.image(masked,width=300,caption="binary segmentation map")

#         # fake = inpaint_unet(masked,binary)
#         st.image(masked,width=600,caption="Inpainted photo")

# elif box=="Training Analysis":
#     fid_frames = []

#     for i in range(1, 3):
#         f = pd.read_csv("FID_epoch" + str(i), header=None)
#         fid_frames.append(f)

#     df = pd.concat(fid_frames)

#     lst = []

#     for i in range(len(df)):
#         if i%2==0:
#             new_row = {"iters": df.iloc[i].values[0], "FID": df.iloc[i + 1].values[0]}
#             lst.append(new_row)

#     dffid_extended = pd.DataFrame(lst, columns=['iters', 'FID'])
#     # Define dffid initially
#     dffid = pd.DataFrame(columns=['iters', 'FID'])
#     # concatenate to original
#     dffid = pd.concat([dffid, dffid_extended])
#     dffid.set_index("iters",inplace=True)

#     st.write("$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$***Frechet Inception Distance (FID)***")
#     st.line_chart(dffid)


#     losses_frames = []

#     for i in range (1,3):
#         f = pd.read_csv("loss_epoch"+str(i),header=None)[1:]
#         losses_frames.append(f)

#     df_l = pd.concat(losses_frames)

#     lst1 = []

#     for i in range(len(df_l)):
#         if i%2==0:
#             new_row = {"iters":float(df.iloc[i].values[0])}
#         if i%2!=0:
#             loss_terms = df_l.iloc[i].values[0].split('    ')
#             new_row["gen"]=float(loss_terms[0])
#             new_row["disc_whole"]=float(loss_terms[1])
#             new_row["disc_mask"]=float(loss_terms[2])
#             new_row["l1_loss"]=float(loss_terms[3])
#             new_row["ssim_loss"]=float(loss_terms[4])
#             new_row["percep"]=float(loss_terms[5])
#         lst1.append(new_row)

#     df_loss = pd.DataFrame(lst1, columns=['iters', 'gen', 'disc_whole', 'disc_mask', 'l1_loss', 'ssim_loss', 'percep'])
#     df_loss.set_index("iters",inplace=True)

#     dfgen = df_loss[["gen"]]
#     st.write("$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$***Generator Loss***")
#     st.line_chart(dfgen)

#     dfdisc = df_loss[["disc_whole","disc_mask"]]
#     st.write("$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$***Discriminators Loss***")
#     st.line_chart(dfdisc)

#     dflosses = df_loss[["l1_loss","ssim_loss","percep"]]
#     st.write("$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$***L1,SSIM,Perceptual Loss***")
#     st.line_chart(dflosses)

import streamlit as st
import time

def inf():
    fps_container = st.empty()
    c1, c2 = st.columns(2)
    i_con = st.empty()
    for i in range(10):
        FPS = i + 1
        with st.container():
            c1.write(f"FPS: {FPS}")
        with c2:
            i_con.write(f"i: {i}")
        time.sleep(1)

def main():
    if st.button("Run Function"):
        inf()

st.markdown(main())

