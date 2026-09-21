import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/transcribe"

st.title("ASR Fusion Demo")

uploaded_file = st.file_uploader(
    "Upload audio",
    type=["wav", "mp3", "m4a"],
)

reference = st.text_area(
    "Reference transcript (optional)"
)

if st.button("Transcribe"):
    if uploaded_file is None:
        st.warning("Please upload an audio file.")
    else:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )
        }

        data = {}

        if reference.strip():
            data["reference"] = reference

        with st.spinner("Processing audio..."):
            response = requests.post(
                API_URL,
                files=files,
                data=data,
            )

        if response.ok:
            result = response.json()

            st.subheader("Transcript")
            st.write(result["transcript"])

            st.subheader("Sentence confidence")

            for item in result["sentence_confidence"]:
                st.write(
                    f'{item["confidence"]:.0f}% — {item["sentence"]}'
                )

            if "evaluation" in result:
                st.subheader("Evaluation")

                st.write(
                    f'WER: {result["evaluation"]["wer"]:.3f}'
                )

                st.write(
                    "Meaning severity:",
                    result["evaluation"]["meaning_score"],
                )

        else:
            st.error(
                f"Request failed: {response.text}"
            )