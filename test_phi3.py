import streamlit as st
from llama_cpp import Llama
import os

# ✅ Correct model path
MODEL_PATH = "C:/models/Phi-3-mini-4k-instruct-q4.gguf"

# ✅ Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found: {MODEL_PATH}")
    st.stop()

# ✅ Load model ONCE in session
if "model" not in st.session_state:
    st.session_state["model"] = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,                 # Large context window
        n_threads=os.cpu_count(),   # Use all CPU cores
        n_batch=2048,               # Increase batch size for faster responses
        use_mlock=True,             # Prevent memory swapping
        n_kv_cache=128,             # Reduce KV Cache to save RAM
        numa=False                  # Avoid NUMA issues
    )

st.title("🦙 Phi-3 Mini Chatbot")

# ✅ User Input
user_input = st.text_input("Ask me anything:", "")

if st.button("Submit"):
    if user_input:
        st.write(f"**You:** {user_input}")

        try:
            # ✅ Generate response
            response_data = st.session_state["model"].create_completion(
                prompt=f"{user_input}\nAI:",
                max_tokens=1024,  # Ensure full answers
                temperature=0.7,
                top_p=0.9,
                stream=True
            )

            response_text = ""
            response_container = st.empty()  # Placeholder for response

            for chunk in response_data:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    choice = chunk["choices"][0]
                    
                    # ✅ Ensure valid text output
                    if "text" in choice:
                        response_text += choice["text"]
                        response_container.markdown(f"**AI:** {response_text}")

                    # ✅ Stop only when response is complete
                    if choice.get("finish_reason") == "stop":
                        break
                else:
                    st.warning("⚠️ Unexpected response format.")

        except Exception as e:
            st.error(f"🚨 Error generating response: {e}")


