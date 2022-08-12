from Recognision_function import *
from PIL import Image, ImageTk
from Wavelet import *
from tkinter import filedialog, messagebox, Tk, Label, Button, CENTER


def openImage():
    global img, image_path, root, uploaded_image, results
    results['text'] = 'Results:\n\n'
    image_path = '\\'.join(filedialog.askopenfilename().split('/'))
    img = Image.open(image_path)
    uploaded_image['text'] = 'Image uploaded!'


def recognise():
    global img, uploaded_image, results, image_path
    if not img:
        messagebox.showinfo('Error', 'Upload image!')
    elif 'dog_' in image_path.split('\\')[-1]:
        orig_img = cv2.imread(image_path)
        signs = signs_crop_image(signs_recognision(find_contours(orig_img)), orig_img)
        signs_names = []
        for sign in signs:
            if find_features((sign)):
                signs_names.append(find_features(sign))
        for i in set(signs_names):
            results['text'] += str(i) + '\n'
    else:
        results['text'] += 'Your image is processing with wavelet...\nWait till the end, please.\n'
        pixels = np.zeros(img.size, dtype=int)
        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                pixels[i, j] = img.getpixel((i, j))[0]
        dog_img = normFactor(grad(dxDOG(pixels), dyDOG(pixels), img))
        results['text'] += 'Finished!\nSigns: \n\n'
        dog_img.convert('RGB')
        orig_img = np.array(dog_img)
        orig_img = orig_img[:, :, ::-1].copy()
        signs = signs_crop_image(signs_recognision(find_contours(orig_img)), orig_img)
        signs_names = []
        for sign in signs:
            if find_features((sign)):
                signs_names.append(find_features(sign))
        if len(signs_names) == 0:
            results['text'] += 'There no signs!'
            return
        for i in set(signs_names):
            results['text'] += str(i) + '\n'


image_path = None
img = Image.new('RGB', (100, 100), 'WHITE')
root = Tk()
root.title('signs recognise')
root.geometry('600x400')
program_name = Label(root, text='Traffic signs recognision', font=('Montserrat', 24), pady=5, padx=150, fg='#323232')
program_name.pack()
uploaded_image = Label(root, text='No image yet', font=('Montserrat', 12), fg='#323232')
uploaded_image.place(relx=0.3, rely=0.5, anchor=CENTER)
upload_btn = Button(root, text='Upload image', font=('Montserrat', 12), fg='#323232', command=openImage)
upload_btn.place(relx=0.3, rely=0.9, anchor=CENTER)
start_btn = Button(root, text='Start', font=('Montserrat', 12), fg='#323232', command=recognise)
start_btn.place(relx=0.7, rely=0.9, anchor=CENTER)
results = Label(root, text='Results: \n\n', font=('Montserrat', 12), fg='#323232')
results.place(relx=0.7, rely=0.6, anchor=CENTER)


def main():
    global root
    root.mainloop()


if __name__ == '__main__':
    main()
