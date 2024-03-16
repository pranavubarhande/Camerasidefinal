import tkinter as tk
from PIL import Image, ImageTk

def quit_window():
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Image, Text, and Button")

# Load and display the image
image = Image.open("test_images/test1.jpg")  # Replace "image.jpg" with the path to your image file
image = image.resize((int(root.winfo_screenwidth() * 0.5), int(root.winfo_screenheight() * 0.9)))
photo = ImageTk.PhotoImage(image)
image_label = tk.Label(root, image=photo)
image_label.grid(row=0, column=0, rowspan=9, sticky="nsew")

# Create and display text
text = "This is the text that will be displayed on the right side."
text_label = tk.Label(root, text=text, wraplength=int(root.winfo_screenwidth() * 0.5))
text_label.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

# Create and display the Quit button
quit_button = tk.Button(root, text="Quit", command=quit_window)
quit_button.grid(row=9, column=0, columnspan=2, sticky="nsew")

# Set row and column weights to make resizing work as intended
root.rowconfigure(0, weight=1)
root.columnconfigure(1, weight=1)

root.mainloop()
