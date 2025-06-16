import streamlit as st
import os
import json

# Page title
st.title("Demo")

# Initialize session state
if "fields" not in st.session_state:
    st.session_state.fields = []

if "show_input" not in st.session_state:
    st.session_state.show_input = False

if "selected_option" not in st.session_state:
    st.session_state.selected_option = None

# Callback to handle field submission
def add_field():
    new_value = st.session_state.get("field_input", "").strip()
    if new_value:
        st.session_state.fields.append(new_value)
    st.session_state.show_input = False  # Immediately hide the input box
    st.session_state.field_input = ""    # Reset the input

# Sidebar controls
with st.sidebar:
    st.header("Upload & Input")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg"])

    # Selectbox
    options = ["Vanilla", "Few Shots", "Self-consistency"]
    selected = st.selectbox("Choose an prompting technqiue", options)
    st.session_state.selected_option = selected

    # Add fields button
    if st.button("Add fields"):
        st.session_state.show_input = True

    # Input shown only when toggled
    if st.session_state.show_input:
        st.text_input(
            "Enter field value",
            key="field_input",
            on_change=add_field,
            label_visibility="visible"
        )

    # Show all added fields
    if st.session_state.fields:
        st.markdown("### Added Fields")
        for i, field in enumerate(st.session_state.fields):
            st.write(f"{i + 1}. {field}")

    # Extract button
    extract_triggered = st.button("Extract")

# Main content
if extract_triggered:
    st.subheader("Extraction Results")
    if uploaded_file is not None:

        image_name = uploaded_file.name
        if os.path.isfile(f"datasets/sroie_v1/images/{image_name}"):
            label_path = f"datasets/sroie_v1/labels/{image_name.split(".")[0]}.json"
            with open(label_path, "r") as f:
                labels = json.load(f)
        else:
            label_path = f"datasets/receipt_vn_v1/labels/{image_name.split(".")[0]}.json"
            with open(label_path, "r") as f:
                labels = json.load(f)

        results = {}

        for field in st.session_state.fields:
            if field in labels:
                results[field] = labels[field]
            else: results[field] = ""

        st.json(results)