import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

st.title("Face App")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
    width:350px
    }

    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
    width:350px
    margin-left:-350px
    }
    </style>
    """,
    unsafe_allow_html=True, )

st.sidebar.title("Setting")
st.sidebar.subheader("Parametrs")

mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh


DEMO_IMAGE = "Face.jpg"
DEMO_VIDEO = "face.mp4"

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpollation=inter)

    return resized


app_mode = st.sidebar.selectbox('Choose the App mode', ['About App', 'Run on image', 'Run on Video'])
if app_mode == 'About App':
    st.markdown("**Nuriddinov Bektosh** hali to'ldirish kerak ")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width:350px
        margin-left:-350px
        }
        </style>
        """,
        unsafe_allow_html=True, )
    #st.video('https:Path')
    """
    Men Haqimda\n
    
    Men bilan bog'lanmoqchi bo'lsangiz:\n
    -[Telegram](Path)\n
    -[Email](Path)\n
    Bizni kuzatib boring\n
    -[Linkden](Path)\n
    -[Kagle](Path)\n
    
    """



elif app_mode == "Run on image":
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    st.sidebar.markdown('---')

    st.markdown("**Detected Faces**")
    kpil_text = st.markdown("0")

    max_face = st.sidebar.number_input("Maximum Number of face", value=2, min_value=1)
    st.sidebar.markdown('-----')
    detection_confidence = st.sidebar.slider("Min Detect Confi", min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('-----')

    img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text("Orginal Images")
    st.sidebar.image(image)

    face_count = 0

    with mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_face,
            min_detection_confidence=detection_confidence) as face_mesh:
        results = face_mesh.process(image)
        out_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            face_count += 1

            mp_drawing.draw_landmarks(
                image=out_image,
                landmark_list=face_landmarks,
                connections=mp_face.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec)
            kpil_text.write(f"<h1 style='text-align:center;color:red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader("Output Image")
        st.image(out_image, use_column_width=True)
if app_mode == "Run on Video":

    st.set_option('deprecation.showfileUploaderEncoding',False)

    use_webcame = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")

    if record:
        st.checkbox("Recording", value=True)
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width:350px
        margin-left:-350px
        }
        </style>
        """,
        unsafe_allow_html=True, )


    max_face = st.sidebar.number_input("Maximum Number of face", value=5, min_value=1)
    st.sidebar.markdown('-----')
    detection_confidence = st.sidebar.slider("Min Detect Confidence", min_value=0.0, max_value=1.0, value=0.5)
    traking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('-----')

    st.markdown("Output")


    stframe = st.empty
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi", "asf", "m4v"])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcame:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    #Recording Part
    codec = cv2.VideoWriter_fourcc('N', 'J', 'P', 'G')

    out = cv2.VideoWriter('output.mp4', codec, fps_input, (width,height))

    st.sidebar.text('Input_Video')
    st.sidebar.video(tffile.name)

    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=1 ,circle_radius=1 )

    kpi1,kpi2,kpi3 = st.beta_columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Img Width**")
        kpi3_text = st.markdown("0")

    st.markdown("hr/>",unsafe_allow_html = True)
    with mp_face.FaceMesh(
        max_num_faces = max_face,
        min_detection_confidence = detection_confidence,) as face_mesh:
        preTime = 0

        while vid.isOpened():
            i+=1
            ret,frame = vid.read()

            if not ret:
                continue
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results =face_mesh.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            face_count=0
            if results.multi_face_landmarks:
                for face_landmark in results.multi_face_landmarks:
                    face_count += 1
                    print(face_landmark)
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmarks_list=face_landmark,
                        connections=mp_face.FACE_CONNECTIONS,
                        landmarks_drawing_spec=drawing_spec,
                        connectRion_drawing_spec=drawing_spec)
            currTime = time.time()
            fps = 1/(currTime-preTime)
            preTime = currTime

            if record:
                out.write(frame)
            kpi1_text.write(f"<h1 style='text-align:center; color:red;'>{int(fps)}</h1>",unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align:center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align:center; color:red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame,(0,0),fx=0.8,fy=0.8)
            frame = image_resize(image=frame,width=640)
            stframe.image(frame,channel="RGB",use_column_width=True)

        st.text("Video Processed")
        output_video = open('output.mp4','rb')
        out_bytes = output_video.read()
        st.video(out_bytes)

        vid.release()
        out.release()