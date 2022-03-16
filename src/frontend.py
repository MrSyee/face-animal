import streamlit as st
from PIL import Image


def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():
    st.title("Upload your Image!")

    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
    if image_file is not None:
        # To See details
        file_details = {
            "filename": image_file.name,
            "filetype": image_file.type,
            "filesize": image_file.size,
        }
        st.write(file_details)

        # To View Uploaded Image
        st.image(load_image(image_file), width=250)


if __name__ == "__main__":
    main()
