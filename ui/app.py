import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import Classifier, DEFAULT_MODEL
from src.database import init_db, get_db
from src.exporter import export_corrections, get_export_stats
from src.vector_store import TAG_LIST


st.set_page_config(page_title="Support Ticket Classifier", layout="wide")


def init_session():
    if "db" not in st.session_state:
        st.session_state.db = init_db()
    if "classifier" not in st.session_state:
        st.session_state.classifier = Classifier()
    if "review_queue" not in st.session_state:
        st.session_state.review_queue = []
    if "history" not in st.session_state:
        st.session_state.history = []


def classify_page():
    st.header("Classify Tickets")

    col1, col2 = st.columns([2, 1])

    with col2:
        mode = st.selectbox("Mode", ["zero_shot", "few_shot", "fine_tuned", "keyword"])
        model = st.text_input("Model", value=DEFAULT_MODEL)

        if st.button("Update Settings"):
            force_fallback = mode == "keyword"
            if mode == "keyword":
                mode = "zero_shot"
            st.session_state.classifier = Classifier(
                model=model, mode=mode, force_fallback=force_fallback
            )
            st.success("Settings updated")

        st.divider()
        health = st.session_state.classifier.check_health()
        if health["status"] == "ready":
            st.success(f"LLM Ready: {health['model']}")
        elif health["status"] == "model_missing":
            st.warning(f"Model missing. Available: {health.get('available', [])}")
        else:
            st.info("Keyword fallback mode (Ollama unavailable)")

    with col1:
        input_type = st.radio(
            "Input Type", ["Single Ticket", "Batch Upload"], horizontal=True
        )

        if input_type == "Single Ticket":
            ticket_text = st.text_area(
                "Ticket Text", height=150, placeholder="Enter support ticket here..."
            )
            ticket_id = st.text_input("Ticket ID (optional)", value="")

            if st.button("Classify", type="primary"):
                if ticket_text.strip():
                    with st.spinner("Classifying..."):
                        result = st.session_state.classifier.classify(
                            ticket_text, ticket_id=ticket_id or None
                        )
                        st.session_state.result = result
                        if result["status"] in ["success", "fallback"]:
                            st.session_state.review_queue.append(result)
                else:
                    st.error("Please enter ticket text")

        else:
            uploaded_file = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])

            if uploaded_file:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        data = json.load(uploaded_file)
                        df = pd.DataFrame(data.get("tickets", data))

                    st.dataframe(df.head())

                    if st.button("Classify All", type="primary"):
                        results = []
                        with st.spinner(f"Classifying {len(df)} tickets..."):
                            for _, row in df.iterrows():
                                result = st.session_state.classifier.classify(
                                    row.get("text", row.get("ticket_text", "")),
                                    ticket_id=row.get("id", row.get("ticket_id", "")),
                                )
                                results.append(result)
                                st.session_state.review_queue.append(result)

                        st.success(f"Classified {len(results)} tickets")
                        st.dataframe(pd.DataFrame(results))

                except Exception as e:
                    st.error(f"Error processing file: {e}")


def review_page():
    st.header("Review Queue")

    queue = st.session_state.review_queue

    if not queue:
        st.info("No tickets to review. Go to Classify to add tickets.")
        return

    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0

    idx = st.session_state.current_idx

    if idx >= len(queue):
        st.success("Queue complete! All tickets reviewed.")
        if st.button("Clear Queue"):
            st.session_state.review_queue = []
            st.session_state.current_idx = 0
        return

    item = queue[idx]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Ticket: {item.get('ticket_id', 'N/A')}")
        st.text_area(
            "Ticket Text",
            item.get("ticket_text", ""),
            height=100,
            disabled=True,
            key=f"text_{idx}",
        )

    with col2:
        st.subheader("AI Predictions")
        for tag in item.get("tags", []):
            st.metric(label=tag["tag"], value=f"{tag['confidence']:.2f}")

        if item.get("reasoning"):
            st.caption(f"Reasoning: {item['reasoning']}")

        st.divider()

        accepted = st.multiselect(
            "Accepted Tags",
            options=TAG_LIST,
            default=[t["tag"] for t in item.get("tags", [])[:1]]
            if item.get("tags")
            else [],
            key=f"accept_{idx}",
        )

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Accept", type="primary"):
                if accepted:
                    db = st.session_state.db
                    db.add_correction(
                        ticket_text=item.get("ticket_text", ""),
                        predicted_tags=[t["tag"] for t in item.get("tags", [])],
                        predicted_confidences=[
                            t["confidence"] for t in item.get("tags", [])
                        ],
                        accepted_tags=accepted,
                        mode=item.get("mode", "zero_shot"),
                        ticket_id=item.get("ticket_id"),
                    )
                st.session_state.current_idx += 1
                st.rerun()

        with col_b:
            if st.button("Skip"):
                st.session_state.current_idx += 1
                st.rerun()

    st.progress((idx + 1) / len(queue), text=f"Ticket {idx + 1} of {len(queue)}")


def history_page():
    st.header("Correction History")

    db = st.session_state.db
    stats = get_export_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Corrections", stats["total_corrections"])
    col2.metric("Min for Export", stats["min_for_export"])
    col3.metric("Ready for Export", "Yes" if stats["ready_for_export"] else "No")

    st.divider()

    corrections = db.get_corrections(limit=100)

    if corrections:
        data = []
        for c in corrections:
            data.append(
                {
                    "ID": c.id,
                    "Ticket ID": c.ticket_id,
                    "Mode": c.mode,
                    "Predicted": json.loads(c.predicted_tags),
                    "Accepted": json.loads(c.accepted_tags),
                    "Confidence Delta": f"{c.confidence_delta:.3f}"
                    if c.confidence_delta
                    else "N/A",
                    "Timestamp": c.timestamp.strftime("%Y-%m-%d %H:%M")
                    if c.timestamp
                    else "N/A",
                }
            )

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

        st.divider()

        if st.button("Export to JSONL", type="primary"):
            result = export_corrections()
            if result["status"] == "success":
                st.success(f"Exported {result['count']} records to {result['path']}")
            else:
                st.warning(result.get("reason", "Export skipped"))
    else:
        st.info("No corrections yet.")


def settings_page():
    st.header("Settings")

    st.subheader("Model Configuration")
    current = st.session_state.classifier

    new_model = st.text_input("Model Name", value=current.model_name)
    new_mode = st.selectbox(
        "Default Mode",
        ["zero_shot", "few_shot", "fine_tuned"],
        index=["zero_shot", "few_shot", "fine_tuned"].index(current.mode),
    )

    if st.button("Save Settings"):
        st.session_state.classifier = Classifier(model=new_model, mode=new_mode)
        st.success("Settings saved")

    st.divider()

    st.subheader("Export")
    export_path = st.text_input("Export Path", value="./data/corrections_export.jsonl")
    min_records = st.number_input(
        "Min Records to Trigger Export", value=10, min_value=1
    )

    if st.button("Export Now"):
        result = export_corrections(output_path=export_path, min_records=min_records)
        if result["status"] == "success":
            st.success(f"Exported {result['count']} records")
        else:
            st.warning(result.get("reason", "Export skipped"))

    st.divider()

    st.subheader("Database")
    db_path = st.text_input("Database Path", value="corrections.db", disabled=True)
    st.caption("Database path is set at initialization")


def main():
    init_session()

    st.title("Support Ticket Auto-Tagger (HITL)")

    page = st.sidebar.radio(
        "Navigation", ["Classify", "Review Queue", "History", "Settings"]
    )

    if page == "Classify":
        classify_page()
    elif page == "Review Queue":
        review_page()
    elif page == "History":
        history_page()
    else:
        settings_page()


if __name__ == "__main__":
    main()
