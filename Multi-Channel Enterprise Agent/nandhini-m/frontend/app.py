import streamlit as st
import sys
import os
import uuid

# =====================================================
# IMPORT PROJECT
# =====================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from main import run_agent

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Pharma Enterprise Agent",
    page_icon="💊",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""

<style>

.main {
    background-color: #F7F9FC;
}

.block-container {
    padding-top: 2rem;
}

.channel-tag {
    font-size: 12px;
    color: #666;
    margin-bottom: 10px;
}

.chat-title {
    font-size: 34px;
    font-weight: 700;
    color: #1F2937;
}

.stChatMessage {
    border-radius: 14px;
    padding: 10px;
}

</style>

""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================

if "session_id" not in st.session_state:

    st.session_state.session_id = (
        f"session_{uuid.uuid4().hex[:6]}"
    )

if "messages" not in st.session_state:

    st.session_state.messages = []

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("💊 Pharma Agent")

    st.caption(
        "Enterprise Assistant"
    )

    st.divider()

    st.markdown(
        f"**Session ID:** `{st.session_state.session_id}`"
    )

    if st.button(
        "🔄 New Session",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.session_id = (
            f"session_{uuid.uuid4().hex[:6]}"
        )

        st.rerun()

# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
<div class='chat-title'>
💊 Pharma Enterprise Agent
</div>
""",
    unsafe_allow_html=True
)

st.markdown(

    f"""
<div class='channel-tag'>

Session:
{st.session_state.session_id}

</div>
""",

    unsafe_allow_html=True
)

st.divider()

# =====================================================
# DEFAULT MESSAGE
# =====================================================

if not st.session_state.messages:

    st.session_state.messages.append({

        "role": "assistant",

        "content":
        """
Hello 👋

I'm your Pharma Enterprise Assistant.

I can help with:

• HR Queries  
• Leave & Payroll  
• Company Policies  
• Compliance Rules  
• Pharma Products  
• Incentives & Sales Targets
"""
    })

# =====================================================
# SHOW CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# =====================================================
# USER INPUT
# =====================================================

user_input = st.chat_input(
    "Ask your question..."
)

# =====================================================
# HANDLE CHAT
# =====================================================

if user_input:

    # =================================================
    # SAVE USER MESSAGE
    # =================================================

    st.session_state.messages.append({

        "role": "user",

        "content": user_input
    })

    # =================================================
    # SHOW USER MESSAGE
    # =================================================

    with st.chat_message("user"):

        st.markdown(user_input)

    # =================================================
    # ASSISTANT RESPONSE
    # =================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            try:

                result = run_agent(

                    session_id=
                    st.session_state.session_id,

                    user_message=
                    user_input
                )

                # =========================================
                # SUCCESS RESPONSE
                # =========================================

                if result["success"]:

                    response = (
                        result["data"]["answer"]
                    )

                # =========================================
                # ERROR RESPONSE
                # =========================================

                else:

                    response = (
                        result["error"]["message"]
                    )

            except Exception as e:

                response = (
                    f"System Error: {str(e)}"
                )

            # =============================================
            # SHOW RESPONSE
            # =============================================

            st.markdown(response)

    # =================================================
    # SAVE ASSISTANT MESSAGE
    # =================================================

    st.session_state.messages.append({

        "role": "assistant",

        "content": response
    })