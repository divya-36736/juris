import streamlit as st
from weather import search_faiss, build_prompt, run_reflexion

st.title("⚖️ Legal Case Advisor")

user_input = st.text_area("Describe your legal issue:")
if st.button("Analyze"):
    similar = search_faiss(user_input)
    prompt = build_prompt(user_input, similar)
    improved = run_reflexion(prompt)

    st.subheader("🔍 Suggested Legal Advice:")
    st.write(improved)
