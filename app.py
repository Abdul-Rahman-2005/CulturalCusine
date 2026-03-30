import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import re
from dotenv import load_dotenv
from groq import Groq

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------
# LOAD ENV VARIABLES
# -------------------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY)

# -------------------------
# STREAMLIT CONFIG
# -------------------------

st.set_page_config(
    page_title="Cultural Cuisine Preservatory",
    layout="wide"
)

# -------------------------
# DATASET SETUP
# -------------------------

DATA_FILE = "cuisine_dataset.csv"

columns = [
    "Dish Name",
    "Region",
    "Ingredients",
    "Cooking Method",
    "Cultural Notes",
    "Preservation Score"
]

if os.path.exists(DATA_FILE):
    try:
        df = pd.read_csv(DATA_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=columns)
else:
    df = pd.DataFrame(columns=columns)

# -------------------------
# SESSION STATE
# -------------------------

if "analysis" not in st.session_state:
    st.session_state.analysis = ""

if "image" not in st.session_state:
    st.session_state.image = None

if "score" not in st.session_state:
    st.session_state.score = None

# -------------------------
# HEADER
# -------------------------

st.title("🍲 Cultural Cuisine Preservatory")

st.markdown("""
AI-powered platform to **preserve traditional cuisines** using AI.

Features:

• Food image upload  
• Cuisine cultural analysis  
• Traditional recipe structuring  
• Cuisine preservation scoring  
• Ingredient analytics dashboard
""")

# -------------------------
# TABS
# -------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Home",
    "Upload Dish",
    "AI Analysis",
    "Visualization",
    "Cultural Report"
])

# -------------------------
# HOME
# -------------------------

with tab1:

    st.subheader("Project Overview")

    st.write("""
This platform helps document **traditional food heritage**.

Users can:

• Upload images of traditional dishes  
• Enter ingredients and cooking methods  
• Generate AI cultural insights  
• Build cuisine preservation datasets
""")

    st.subheader("Dataset")

    st.dataframe(df)

# -------------------------
# IMAGE PREPROCESSING (FIXED)
# -------------------------

def preprocess_image(uploaded_file):

    image = Image.open(uploaded_file)

    # Resize using PIL (instead of cv2)
    resized = image.resize((240, 320))

    # Convert to grayscale using PIL
    gray = resized.convert("L")

    return resized, gray

# -------------------------
# INPUT TAB
# -------------------------

with tab2:

    st.subheader("Upload Traditional Dish")

    uploaded_image = st.file_uploader(
        "Upload Food Image",
        type=["jpg","png","jpeg"]
    )

    if uploaded_image:

        image, processed = preprocess_image(uploaded_image)

        st.session_state.image = image

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", width=240)

        with col2:
            st.image(processed, caption="Processed Image", width=240)

    st.markdown("### Enter Dish Details")

    dish_name = st.text_input("Dish Name")
    region = st.text_input("Region / Country")

    ingredients = st.text_area(
        "Ingredients (comma separated)"
    )

    cooking_method = st.text_area("Cooking Method")

    cultural_notes = st.text_area("Cultural Notes (Optional)")

# -------------------------
# AI ANALYSIS FUNCTION
# -------------------------

def run_ai_analysis():

    prompt = f"""
Analyze the following traditional cuisine.

Dish Name: {dish_name}
Region: {region}
Ingredients: {ingredients}
Cooking Method: {cooking_method}
Cultural Notes: {cultural_notes}

Provide clearly formatted sections:

1. Cuisine Identification
2. Ingredient Cultural Analysis
3. Cultural Significance
4. Structured Traditional Recipe
5. Cuisine Preservation Score (0-100)

For structured recipe include:

Ingredient List
Step-by-step Instructions
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    result = completion.choices[0].message.content

    score_match = re.search(r'(\d{2,3})', result)
    score = int(score_match.group()) if score_match else 75

    return result, score

# -------------------------
# AI ANALYSIS TAB
# -------------------------

with tab3:

    st.subheader("AI Cultural Cuisine Analysis")

    if st.button("Run AI Analysis"):

        if dish_name == "" or ingredients == "" or st.session_state.image is None:
            st.warning("Please upload image and enter dish details.")

        else:

            with st.spinner("Analyzing cuisine with AI..."):

                result, score = run_ai_analysis()

                st.session_state.analysis = result
                st.session_state.score = score

                st.success("Analysis Complete")

                st.write(result)

                st.write(f"### Cuisine Preservation Score: {score}/100")

                new_row = pd.DataFrame([{
                    "Dish Name": dish_name,
                    "Region": region,
                    "Ingredients": ingredients,
                    "Cooking Method": cooking_method,
                    "Cultural Notes": cultural_notes,
                    "Preservation Score": score
                }])

                df = pd.concat([df, new_row], ignore_index=True)

                df.to_csv(DATA_FILE, index=False)

# -------------------------
# VISUALIZATION TAB
# -------------------------

with tab4:

    st.subheader("Cuisine Analytics Dashboard")

    if not df.empty:

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Region Wise Cuisine Consumption")

            region_counts = df["Region"].value_counts()

            fig1, ax1 = plt.subplots(figsize=(6,4))

            region_counts.plot(kind="bar", ax=ax1)

            ax1.set_xlabel("Region")
            ax1.set_ylabel("Number of Dishes")
            ax1.set_title("Region Wise Cuisine Distribution")

            st.pyplot(fig1)

        with col2:

            st.markdown("### Ingredient Quantity Distribution")

            ingredient_series = df["Ingredients"].str.split(",").explode()

            ingredient_counts = ingredient_series.value_counts().head(8)

            fig2, ax2 = plt.subplots(figsize=(6,4))

            ax2.pie(
                ingredient_counts,
                labels=ingredient_counts.index,
                autopct="%1.1f%%"
            )

            ax2.set_title("Ingredient Distribution")

            st.pyplot(fig2)

    else:

        st.info("No cuisine data available yet.")

# -------------------------
# PDF GENERATOR
# -------------------------

def generate_pdf():

    styles = getSampleStyleSheet()
    pdf_file = "cuisine_report.pdf"
    story = []

    story.append(Paragraph("Cultural Cuisine Preservation Report", styles['Title']))
    story.append(Spacer(1,20))

    story.append(Paragraph(f"Dish Name: {dish_name}", styles['Normal']))
    story.append(Paragraph(f"Region: {region}", styles['Normal']))
    story.append(Spacer(1,10))

    story.append(Paragraph("Ingredients:", styles['Heading3']))
    story.append(Paragraph(ingredients, styles['Normal']))
    story.append(Spacer(1,10))

    story.append(Paragraph("Cooking Method:", styles['Heading3']))
    story.append(Paragraph(cooking_method, styles['Normal']))
    story.append(Spacer(1,10))

    story.append(Paragraph("Cultural Notes:", styles['Heading3']))
    story.append(Paragraph(cultural_notes, styles['Normal']))
    story.append(Spacer(1,10))

    if st.session_state.image is not None:

        image_path = "dish_image.png"
        st.session_state.image.save(image_path)

        story.append(Spacer(1,10))
        story.append(RLImage(image_path, width=240, height=320))
        story.append(Spacer(1,15))

    story.append(Paragraph("AI Cultural Analysis", styles['Heading2']))
    story.append(Paragraph(st.session_state.analysis, styles['Normal']))

    story.append(Spacer(1,20))

    story.append(Paragraph(
        f"Cuisine Preservation Score: {st.session_state.score}/100",
        styles['Heading3']
    ))

    doc = SimpleDocTemplate(pdf_file)
    doc.build(story)

    return pdf_file

# -------------------------
# CULTURAL REPORT
# -------------------------

with tab5:

    st.subheader("Cultural Preservation Report")

    if st.session_state.analysis != "" and st.session_state.image is not None:

        st.image(st.session_state.image, width=240)

        st.markdown("### AI Cultural Insights")

        st.write(st.session_state.analysis)

        pdf_path = generate_pdf()

        with open(pdf_path,"rb") as f:

            st.download_button(
                "Download Full Report (PDF)",
                f,
                file_name="cuisine_report.pdf",
                mime="application/pdf"
            )

    else:

        st.info("Upload dish image and run AI analysis first.")
