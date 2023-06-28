import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model

model = load_model('./model.h5')
# dicionário para rotular todas as classes de câncer de pulmão
classes = {
    0: 'Classificação: Adenocarcinoma',
    1: 'Classificação: Large Cell Carcinoma',
    2: 'Classificação: Normal',
    3: 'Classificação: Squamous Cell Carcinoma',
}
# inicializa a GUI
top = tk.Tk()
top.geometry('1920x1080')
top.title('PulmoScan')
imagem = Image.open("./public/Logotipo.png")
imagem_tk = ImageTk.PhotoImage(imagem)
background = Image.open("./public/background.png")
background_tk = ImageTk.PhotoImage(background)
label_background = tk.Label(top, image=background_tk)
label_background.place(x=0, y=0, relwidth=1, relheight=1)
top.configure(background='#FFF')
label = Label(top,
              background='white',
              font=('arial', 20, 'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.convert("RGB")
    image = image.resize((299, 299))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0
    pred_probs = model.predict(image)[0]
    pred = np.argmax(pred_probs)
    sign = classes[int(pred)]
    print(sign)
    label.configure(foreground='#00E3F0', text=sign)


def show_classify_button(file_path):
    classify_b = Button(
        top,
        text="Classify Image",
        command=lambda: classify(file_path),
        padx=10,
        pady=5
    )
    classify_b.configure(
        background='#00E3F0',
        foreground='white',
        font=('arial', 10, 'bold')
    )
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25),
                           (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(
    top,
    text="Upload an image",
    command=upload_image,
    padx=10,
    pady=5
)
upload.configure(
    background='#00E3F0',
    foreground='white',
    font=('arial', 10, 'bold')
)
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(
    top,
    image=imagem_tk,
    pady=20,
    font=('arial', 20, 'bold')
)
heading.configure(background='#FFF', foreground='#00E3F0')
heading.pack()
top.mainloop()
