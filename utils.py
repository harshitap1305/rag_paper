import streamlit as st

def display_image(image_data, caption="", width=None):
    """Display image with proper Streamlit compatibility"""
    try:
        st.image(image_data, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(image_data, caption=caption, use_column_width=True)
        except TypeError:
            st.image(image_data, caption=caption, width=width or 700)