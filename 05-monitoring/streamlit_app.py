from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from assistant import create_assistant
from db_feedback import save_feedback
from db_save import save_conversation
from judge import evaluate_relevance, print_relevance, should_run_judge

st.set_page_config(page_title="Course Assistant", page_icon="🎓", layout="wide")


@st.cache_resource
def get_assistant():
    return create_assistant()


@st.cache_resource
def get_judge_executor():
    return ThreadPoolExecutor(max_workers=2)


def start_judge_evaluation(query, answer):
    st.session_state.judge_result = None
    st.session_state.judge_future = get_judge_executor().submit(
        evaluate_relevance, query, answer
    )


@st.fragment(run_every=1)
def show_judge_result():
    future = st.session_state.get("judge_future")
    if future is None or future.done() is False:
        if future is not None:
            st.caption("Evaluating relevance...")
        return

    if st.session_state.get("judge_saved"):
        result = st.session_state.judge_result
        print_relevance(result['relevance'], result['explanation'], st)
        return

    try:
        relevance, explanation = future.result()
    except Exception as exc:
        st.error(f"Judge evaluation failed: {exc}")
        st.session_state.judge_future = None
        return

    save_feedback(
        st.session_state.conversation_id,
        "judge",
        relevance=relevance,
        explanation=explanation,
    )
    st.session_state.judge_result = {
        "relevance": relevance,
        "explanation": explanation,
    }
    st.session_state.judge_saved = True
    print_relevance(relevance, explanation, st)


def render_assistant():
    st.title("Assistant")
    assistant = get_assistant()

    query = st.text_input("Enter your question")

    if st.button("Ask"):
        with st.spinner("Processing..."):
            answer = assistant.rag(query)
            record = assistant.last_call_record
            conversation_id = save_conversation(record, query, "llm-zoomcamp")

            st.session_state.query = query
            st.session_state.answer = answer
            st.session_state.conversation_id = conversation_id
            st.session_state.record = record
            st.session_state.feedback_message = None
            st.session_state.judge_saved = False
            st.session_state.judge_result = None
            st.session_state.judge_future = None
            if should_run_judge():
                start_judge_evaluation(query, answer)

    if st.session_state.get("answer"):
        st.success("Completed!")
        st.write(st.session_state.answer)

        with st.container(horizontal=True):
            if st.button("+1", key="feedback_plus"):
                save_feedback(st.session_state.conversation_id, "user", score=1)
                st.session_state.feedback_message = "Thanks!"

            if st.button("-1", key="feedback_minus"):
                save_feedback(st.session_state.conversation_id, "user", score=-1)
                st.session_state.feedback_message = "Thanks for the feedback!"

        if st.session_state.get("feedback_message"):
            st.write(st.session_state.feedback_message)

        record = st.session_state.record
        st.write(f"Response time: {record.response_time:.2f}s")
        st.write(f"Prompt tokens: {record.prompt_tokens}")
        st.write(f"Completion tokens: {record.completion_tokens}")
        st.write(f"Cost: ${record.cost:.6f}")
        st.write(f"Timestamp: {record.timestamp:.2f}")

        show_judge_result()


pg = st.navigation(
    [
        st.Page(render_assistant, title="Assistant", icon="💬"),
        st.Page("dashboard.py", title="Dashboard", icon="📊"),
    ],
    position="top",
)
pg.run()
